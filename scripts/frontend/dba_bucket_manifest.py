import json
from dataclasses import asdict, dataclass
from pathlib import Path


SUPPORTED_FRONTEND_IMAGE_SIZE = (344, 616)
SUPPORTED_FEATURE_SHAPE = (43, 77)
SUPPORTED_MODE = "vo"
SUPPORTED_BACKEND = "torch"


@dataclass(frozen=True)
class DBACallSignature:
    active_edges: int
    ba_edges: int
    source_poses: int
    pose_window: int
    use_inactive: bool
    upsample: bool
    dtype: str
    feature_shape: tuple
    frontend_image_size: tuple
    mode: str
    backend: str

    def support_error(self):
        if tuple(self.frontend_image_size) != SUPPORTED_FRONTEND_IMAGE_SIZE:
            return "frontend_image_size"
        if tuple(self.feature_shape) != SUPPORTED_FEATURE_SHAPE:
            return "feature_shape"
        if self.mode != SUPPORTED_MODE:
            return "mode"
        if self.backend != SUPPORTED_BACKEND:
            return "backend"
        if self.dtype not in {"float16", "float32"}:
            return "dtype"
        return None

    def to_dict(self):
        result = asdict(self)
        result["feature_shape"] = list(self.feature_shape)
        result["frontend_image_size"] = list(self.frontend_image_size)
        return result


@dataclass(frozen=True)
class DBABucketSpec:
    name: str
    active_edges: int
    ba_edges: int
    source_poses: int
    pose_window: int
    use_inactive: bool
    upsample: bool
    dtype: str
    max_padding_ratio: float
    estimated_workspace_bytes: int

    def padding_ratio(self, call):
        capacity = (
            self.active_edges
            + self.ba_edges
            + self.source_poses
            + self.pose_window
        )
        padding = (
            self.active_edges
            - call.active_edges
            + self.ba_edges
            - call.ba_edges
            + self.source_poses
            - call.source_poses
            + self.pose_window
            - call.pose_window
        )
        return padding / capacity

    def fits(self, call):
        if call.support_error() is not None:
            return False
        if (
            self.use_inactive != call.use_inactive
            or self.upsample != call.upsample
            or self.dtype != call.dtype
        ):
            return False
        if (
            call.active_edges > self.active_edges
            or call.ba_edges > self.ba_edges
            or call.source_poses > self.source_poses
            or call.pose_window > self.pose_window
        ):
            return False
        return self.padding_ratio(call) <= self.max_padding_ratio


@dataclass(frozen=True)
class DBABucketManifest:
    schema_version: int
    frontend_image_size: tuple
    feature_shape: tuple
    mode: str
    backend: str
    buckets: tuple

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        image_size = tuple(payload["frontend_image_size"])
        feature_shape = tuple(payload["feature_shape"])
        if (
            image_size != SUPPORTED_FRONTEND_IMAGE_SIZE
            or feature_shape != SUPPORTED_FEATURE_SHAPE
        ):
            raise ValueError("DBA bucket manifest must target 344x616 -> 43x77")
        if (
            payload["mode"] != SUPPORTED_MODE
            or payload["backend"] != SUPPORTED_BACKEND
        ):
            raise ValueError(
                "DBA bucket manifest must target VO with the torch backend"
            )
        buckets = tuple(DBABucketSpec(**item) for item in payload["buckets"])
        return cls(
            schema_version=int(payload["schema_version"]),
            frontend_image_size=image_size,
            feature_shape=feature_shape,
            mode=payload["mode"],
            backend=payload["backend"],
            buckets=buckets,
        )

    def select(self, call):
        candidates = [bucket for bucket in self.buckets if bucket.fits(call)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda bucket: (
                bucket.estimated_workspace_bytes,
                bucket.active_edges,
                bucket.ba_edges,
                bucket.name,
            ),
        )
