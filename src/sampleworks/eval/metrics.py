from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float
from loguru import logger


def rscc(array1, array2):
    """Calculate the Real Space Correlation Coefficient between two arrays.

    Returns NaN if correlation cannot be computed.
    """
    if array1.shape != array2.shape:
        # FIXME? should this raise an error @karson
        logger.warning(f"Shape mismatch: {array1.shape} vs {array2.shape}")
        return np.nan

    if array1.size == 0 or array2.size == 0:
        logger.warning("Empty array provided to rscc")
        return np.nan

    # Flatten arrays
    arr1_flat = array1.flatten()
    arr2_flat = array2.flatten()

    # Check for NaN/Inf
    # TODO: q for Karson: do you want to ignore these values instead?
    if not (np.isfinite(arr1_flat).all() and np.isfinite(arr2_flat).all()):
        logger.warning("NaN or Inf values in input arrays")
        return np.nan

    # Check for zero variance (constant arrays)
    if np.std(arr1_flat) < 1e-10 or np.std(arr2_flat) < 1e-10:
        logger.warning("Zero or near-zero variance in input arrays")
        return np.nan

    try:
        corr = np.corrcoef(arr1_flat, arr2_flat)[0, 1]
        return corr
    except Exception as e:
        logger.warning(f"Correlation calculation failed: {e}")
        return np.nan


@dataclass(frozen=True)
class RFactors:
    """Crystallographic R factors of one model against one set of observed amplitudes.

    Attributes
    ----------
    work
        R over the working set.
    free
        R over the test set; ``None`` when no reflection is flagged free.
    overall
        R over every reflection.
    """

    work: float
    free: float | None
    overall: float


def r_factors(
    f_obs: Float[torch.Tensor, " n_refl"],
    f_calc: Float[torch.Tensor, " n_refl"],
    free_mask: Bool[torch.Tensor, " n_refl"],
    outlier_mask: Bool[torch.Tensor, " n_refl"],
) -> RFactors:
    """Compute R-work, R-free and R-all, ``R = sum||Fo| - |Fc|| / sum|Fo|`` without
    scale fitting.

    Parameters
    ----------
    f_obs
        Observed amplitudes ``|Fo|``, ``[n_refl]``.
    f_calc
        Model amplitudes ``|Fc|``, ``[n_refl]``.
    free_mask
        True where the reflection is in the R-free test set, ``[n_refl]``.
    outlier_mask
        True where the reflection is left out of every R factor, ``[n_refl]``.

    Returns
    -------
    RFactors
        The three R factors.

    Raises
    ------
    ValueError
        If the four tensors are not one-dimensional over the same reflections, or the
        working set is empty.
    """
    shapes = {tuple(t.shape) for t in (f_obs, f_calc, free_mask, outlier_mask)}
    if len(shapes) != 1 or len(next(iter(shapes))) != 1:
        raise ValueError(
            "f_obs, f_calc, free_mask and outlier_mask must be one-dimensional over the "
            f"same reflections; got shapes {sorted(shapes)}"
        )
    work = ~free_mask & ~outlier_mask
    free = free_mask & ~outlier_mask
    if not work.any():
        raise ValueError("No reflection is neither free nor an outlier; the working set is empty")
    residual = (f_obs - f_calc).abs()

    def r_factor(selection: torch.Tensor) -> float:
        """R over the reflections where boolean ``selection`` ``[n_refl]`` is True."""
        return float(residual[selection].sum() / f_obs[selection].sum())

    return RFactors(
        work=r_factor(work),
        free=r_factor(free) if free.any() else None,
        overall=r_factor(~outlier_mask),
    )
