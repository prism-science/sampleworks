import os
import sys
from pathlib import Path

from torch.utils.cpp_extension import load


def _ensure_toolchain_env() -> None:
    """Ensure torch extensions pick the right toolchain.

    When Python is launched directly (e.g., for marimo) it does not run
    through ``pixi run`` so environment variables like CC/CXX are unset.
    Torch falls back to the system g++, which may be too old
    for PyTorch kernels. This derives the conda toolchain that lives next to
    the active interpreter and export it on-demand so every execution path
    (scripts, notebooks, marimo) uses the supported compiler versions.
    """

    bin_dir = Path(sys.executable).resolve().parent
    cc = bin_dir / "x86_64-conda-linux-gnu-cc"
    cxx = bin_dir / "x86_64-conda-linux-gnu-c++"

    if cc.exists() and not os.environ.get("CC"):
        os.environ["CC"] = str(cc)

    if cxx.exists() and not os.environ.get("CXX"):
        os.environ["CXX"] = str(cxx)


_ensure_toolchain_env()

_this_dir = os.path.dirname(os.path.abspath(__file__))

try:
    # Try to load the compiled extension
    dilate_points_cuda = load(
        name="dilate_points_cuda",
        sources=[
            os.path.join(_this_dir, "dilate_points_ext.cpp"),
            os.path.join(_this_dir, "cuda", "dilate_points_impl.cu"),
            os.path.join(_this_dir, "cuda", "dilate_points_kernel.cu"),
        ],
        verbose=True,
    )
    CUDA_AVAILABLE = True
    CUDA_LOAD_ERROR = None
except Exception as e:
    print(f"CUDA extension loading failed: {e}")
    dilate_points_cuda = None
    CUDA_AVAILABLE = False
    CUDA_LOAD_ERROR = f"{type(e).__name__}: {e}"
