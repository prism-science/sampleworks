"""Tests for choosing between the CUDA density kernel and the PyTorch density path."""

import pytest
import torch
from sampleworks.core.forward_models.xray.real_space_density import use_cuda_density_kernels
from sampleworks.utils.guidance_script_arguments import GuidanceConfig


BUILD_ERROR = "RuntimeError: Ninja is required to load C++ extensions"
CUDA = torch.device("cuda")


@pytest.mark.parametrize("allow_torch_fallback", [False, True])
@pytest.mark.parametrize("kernel_load_error", [None, BUILD_ERROR])
def test_cpu_always_uses_the_pytorch_path(allow_torch_fallback, kernel_load_error):
    """The kernel is CUDA-only, so a CPU device never asks for it, whatever the build state.

    Parameters
    ----------
    allow_torch_fallback
        Whether fallback is allowed.
    kernel_load_error
        Build error, or None for a built kernel.
    """
    assert not use_cuda_density_kernels(
        torch.device("cpu"),
        allow_torch_fallback=allow_torch_fallback,
        kernel_load_error=kernel_load_error,
    )


@pytest.mark.parametrize("allow_torch_fallback", [False, True])
def test_cuda_uses_a_built_kernel(allow_torch_fallback):
    """A built kernel is used on CUDA whether or not fallback is allowed.

    Parameters
    ----------
    allow_torch_fallback
        Whether fallback is allowed.
    """
    assert use_cuda_density_kernels(
        CUDA, allow_torch_fallback=allow_torch_fallback, kernel_load_error=None
    )


def test_cuda_without_a_built_kernel_fails_by_default():
    """The error names the build failure and the option that permits the PyTorch path."""
    with pytest.raises(RuntimeError, match="Ninja is required.*--allow-torch-fallback"):
        use_cuda_density_kernels(CUDA, kernel_load_error=BUILD_ERROR)


def test_cuda_without_a_built_kernel_falls_back_when_allowed():
    """With fallback allowed, a missing kernel selects the PyTorch path and warns."""
    with pytest.warns(UserWarning, match="PyTorch density path"):
        assert not use_cuda_density_kernels(
            CUDA, allow_torch_fallback=True, kernel_load_error=BUILD_ERROR
        )


@pytest.mark.parametrize(["flags", "expected"], [[[], False], [["--allow-torch-fallback"], True]])
def test_fallback_is_a_density_loss_option(tmp_path, flags, expected):
    """``--allow-torch-fallback`` reaches the density loss's options, and is off by default.

    Parameters
    ----------
    tmp_path
        Folder holding a placeholder map; the CLI only checks that the file exists.
    flags
        Extra command-line flags.
    expected
        The option's resulting value.
    """
    density_path = tmp_path / "1vme.ccp4"
    density_path.touch()
    config = GuidanceConfig.from_cli(
        [
            "--model",
            "boltz2",
            "--guidance-type",
            "pure_guidance",
            "--protein",
            "1VME",
            "--structure",
            "1vme.cif",
            "--density",
            str(density_path),
            "--resolution",
            "1.8",
            *flags,
        ]
    )
    (entry,) = config.resolved_loss_config().with_effective_options().entries

    assert entry.options["allow_torch_fallback"] is expected
