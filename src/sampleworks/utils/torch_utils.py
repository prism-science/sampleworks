"""
Utility functions for working with PyTorch tensors and devices.
"""

from typing import Any

import numpy as np
import torch

from sampleworks import should_check_nans


DeviceLikeType = str | torch.device | int


def try_gpu():
    """Attempt to select the free-est GPU (by memory) for use with PyTorch.

    Returns
    -------
    torch.device
        GPU with the most available memory, or the CPU if no GPUs are available.
    """
    import os
    import subprocess
    from io import StringIO

    import pandas as pd

    try:
        gpu_stats = subprocess.check_output(
            ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
        )
        gpu_stats_str = gpu_stats.decode("utf-8")
        gpu_df = pd.read_csv(
            StringIO(gpu_stats_str), names=["memory.used", "memory.free"], skiprows=1
        )
        print(f"GPU usage:\n{gpu_df}")
        gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
        if gpu_df.empty:
            print("No GPUs found.")
            return torch.device("cpu")
        idx = gpu_df["memory.free"].idxmax()

        available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if available_gpus is not None:
            available_gpus = [int(gpu) for gpu in available_gpus.split(",")]
            if idx not in available_gpus:
                print(
                    f"Selected GPU {idx} is not in CUDA_VISIBLE_DEVICES: \
                    {available_gpus}. Selecting first available GPU: \
                    {available_gpus[0]}."
                )
                return torch.device("cuda:0")

            # Adjust idx to match the actual GPU index, which gets remapped by CUDA_VISIBLE_DEVICES
            # e.g. if CUDA_VISIBLE_DEVICES=2,3 and idx=3 (the 3rd GPU according to nvidia-smi),
            # we need to return cuda:1 (the 2nd GPU in the visible list)
            idx = available_gpus.index(idx)

        print(
            "Returning GPU{} with {} free MiB".format(
                idx,
                gpu_df.iloc[idx]["memory.free"],
            )
        )

        return torch.device(f"cuda:{idx}")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Failed to run nvidia-smi: {e}")
        return torch.device("cpu")


def send_tensors_in_dict_to_device(d: dict, device: DeviceLikeType, inplace: bool = True) -> dict:
    """Recursively sends all torch.Tensors in a dictionary to a specified device.

    Parameters
    ----------
    d: dict
        The input dictionary potentially containing torch.Tensors.
    device: str | torch.device | int
        The target device to send the tensors to (e.g., 'cpu', 'cuda:0').
    inplace: bool, optional
        If True, modifies the input dictionary in place. If False,
        returns a new dictionary.
        Default is True.

    Returns
    -------
    dict
        The dictionary with all torch.Tensors sent to the specified device.
    """
    result = d if inplace else {}
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, dict):
            result[key] = send_tensors_in_dict_to_device(value, device=device, inplace=inplace)
        elif isinstance(value, list | tuple):
            result[key] = type(value)(
                send_tensors_in_dict_to_device(item, device=device, inplace=inplace)
                if isinstance(item, dict)
                else item.to(device)
                if isinstance(item, torch.Tensor)
                else item
                for item in value
            )
        else:
            result[key] = value
    return result


def _assert_no_nans(x: Any, *, msg: str = "", fail_if_not_tensor: bool = False) -> None:
    """Recursively checks for NaN values in tensor-like objects.

    Parameters
    ----------
    x : Any
        Input to check for NaNs. Can be a tensor, dict, list, tuple, or other type.
    msg : str
        Prefix for error messages.
    fail_if_not_tensor : bool
        If True, raises error for non-tensor types.
    """
    if isinstance(x, torch.Tensor):
        torch._assert(
            not torch.isnan(x).any(),
            ": ".join(filter(bool, [msg, "Tensor contains NaNs!"])),
        )
    elif isinstance(x, np.ndarray):
        torch._assert(
            not np.isnan(x).any(),
            ": ".join(filter(bool, [msg, "Numpy array contains NaNs!"])),
        )
    elif isinstance(x, float):
        torch._assert(
            not np.isnan(x),
            ": ".join(filter(bool, [msg, "float is NaN!"])),
        )
    elif isinstance(x, dict):
        for k, v in x.items():
            _assert_no_nans(
                v,
                msg=".".join(filter(bool, [msg, k])),
                fail_if_not_tensor=fail_if_not_tensor,
            )
    elif isinstance(x, (list, tuple)):
        for idx, v in enumerate(x):
            _assert_no_nans(
                v,
                msg=".".join(filter(bool, [msg, str(idx)])),
                fail_if_not_tensor=fail_if_not_tensor,
            )
    elif fail_if_not_tensor:
        raise ValueError(f"Unsupported type: {type(x)}")


def do_nothing(*args: Any, **kwargs: Any) -> None:
    """Does nothing, just returns None"""
    pass


assert_no_nans = _assert_no_nans if should_check_nans else do_nothing


def resize_to_ensemble(tensor: torch.Tensor, ensemble_size: int) -> torch.Tensor:
    """Resize a tensor to the specified ensemble size by repeating the first dimension.

    Parameters
    ----------
    tensor
        Input tensor to resize
    ensemble_size
        Target ensemble size for the first dimension

    Returns
    -------
    torch.Tensor
        Tensor with first dimension expanded to ensemble_size
    """
    if tensor.ndim < 2:
        tensor = tensor.unsqueeze(0)
    return tensor.repeat(ensemble_size, *[1] * (tensor.ndim - 1))


def gaussian_prior(
    batch_size: int,
    num_atoms: int | None,
    device: DeviceLikeType,
    *,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Draw standard Gaussian coordinates, the prior of the flow model wrappers.

    Parameters
    ----------
    batch_size : int
        Number of samples to draw.
    num_atoms : int | None
        Atom count taken from the model's features, or ``None`` when no features are
        available.
    device : DeviceLikeType
        Device on which to allocate the samples.
    shape : tuple[int, ...] | None, optional
        Explicit ``(num_atoms, 3)`` shape of one sample. Overrides ``num_atoms``.

    Returns
    -------
    torch.Tensor
        Gaussian coordinates, shape ``(batch_size, atoms, 3)``.

    Raises
    ------
    ValueError
        If ``shape`` is not of the form ``(num_atoms, 3)``, or if both ``shape`` and
        ``num_atoms`` are ``None``.
    """
    if shape is not None:
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError("shape must be of the form (num_atoms, 3)")
        return torch.randn((batch_size, *shape), device=device)
    if num_atoms is None:
        raise ValueError("Either features or shape must be provided to initialize_from_prior()")
    return torch.randn((batch_size, num_atoms, 3), device=device)
