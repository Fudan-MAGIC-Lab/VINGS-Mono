from pathlib import Path
import shlex
import subprocess

import numpy as np
from setuptools import Extension, setup
from torch.utils.cpp_extension import include_paths as torch_include_paths

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "scripts" / "frontend" / "vpi_ofa_gate_cpp.cpp"

gst_cflags = shlex.split(
    subprocess.check_output(
        ["pkg-config", "--cflags", "gstreamer-1.0", "gstreamer-app-1.0", "gstreamer-video-1.0"],
        text=True,
    )
)
gst_libs = shlex.split(
    subprocess.check_output(
        ["pkg-config", "--libs", "gstreamer-1.0", "gstreamer-app-1.0", "gstreamer-video-1.0"],
        text=True,
    )
)
gst_include_dirs = [arg[2:] for arg in gst_cflags if arg.startswith("-I")]
gst_extra_compile_args = [arg for arg in gst_cflags if not arg.startswith("-I")]
gst_library_dirs = [arg[2:] for arg in gst_libs if arg.startswith("-L")]
gst_libraries = [arg[2:] for arg in gst_libs if arg.startswith("-l")]
gst_extra_link_args = [arg for arg in gst_libs if not arg.startswith("-L") and not arg.startswith("-l")]

extension = Extension(
    "vpi_ofa_gate_cpp",
    sources=[str(SRC)],
    include_dirs=[
        *torch_include_paths(),
        np.get_include(),
        "/opt/nvidia/vpi2/include",
        "/usr/include/opencv4",
        "/usr/src/jetson_multimedia_api/include",
        *gst_include_dirs,
    ],
    library_dirs=[
        "/opt/nvidia/vpi2/lib/aarch64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        *gst_library_dirs,
    ],
    libraries=["nvvpi", "opencv_core", "opencv_imgproc", *gst_libraries],
    extra_compile_args=["-O3", "-std=c++17", *gst_extra_compile_args],
    extra_link_args=["-Wl,-rpath,/opt/nvidia/vpi2/lib/aarch64-linux-gnu", *gst_extra_link_args],
)

setup(
    name="vpi_ofa_gate_cpp",
    ext_modules=[extension],
)
