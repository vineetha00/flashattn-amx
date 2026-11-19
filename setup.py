import platform
import sys

from setuptools import Extension, setup

if sys.platform != "darwin" or platform.machine() != "arm64":
    sys.exit(
        "flashattn-amx requires macOS on Apple Silicon: the kernel runs on the "
        "M-series AMX matrix coprocessor via Apple's Accelerate framework."
    )

setup(
    ext_modules=[
        Extension(
            "flashattn_amx._core",
            sources=["src/flashattn_amx/_core.c"],
            extra_compile_args=["-O3", "-DACCELERATE_NEW_LAPACK"],
            extra_link_args=["-framework", "Accelerate"],
        )
    ]
)
