import unittest
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


class DBAFusionMotionGatePacketTests(unittest.TestCase):
    def test_track_passes_full_data_packet_to_motion_gate_when_supported(self):
        from scripts.frontend.dbaf import DBAFusion

        calls = []

        class FakeGate:
            def decide_packet(self, packet):
                calls.append(packet)
                return {"skip": True, "score": 0.0, "reason": "low_motion", "backend": "vpi_nvbuffer"}

        class FakeVideo:
            pass

        fusion = object.__new__(DBAFusion)
        fusion.motion_gate = FakeGate()
        fusion.video = FakeVideo()
        fusion.profiler = None
        packet = {"timestamp": 0, "rgb": object(), "intrinsic": object(), "motion_gate_dmabuf_fd": 3}

        fusion.track(packet)

        self.assertEqual(calls, [packet])
        self.assertEqual(fusion.video.last_motion_gate_backend, "vpi_nvbuffer")


if __name__ == "__main__":
    unittest.main()
