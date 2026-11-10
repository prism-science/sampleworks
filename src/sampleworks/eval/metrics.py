from dataclasses import dataclass

import numpy as np
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
        R over the working set (reflections not flagged free).
    free
        R over the test set; ``None`` when no reflection is flagged free.
    all
        R over every reflection. The number to compare when guidance saw every reflection,
        where the work/free split is two arbitrary subsets rather than seen versus held out.
    """

    work: float
    free: float | None
    all: float


def r_factors(
    f_obs: Float[np.ndarray, " n_refl"],
    f_calc: Float[np.ndarray, " n_refl"],
    free: Bool[np.ndarray, " n_refl"],
) -> RFactors:
    """Compute R-work, R-free and R-all, ``R = sum||Fo| - |Fc|| / sum|Fo|``, with no scale fit.

    Unscaled, so ``f_obs`` and ``f_calc`` must already be on one absolute scale, as synthetic
    targets and their forward model are. The engine that produced ``f_calc`` does not
    matter; every array must be indexed by the same reflections.

    Parameters
    ----------
    f_obs
        Observed amplitudes ``|Fo|``.
    f_calc
        Model amplitudes ``|Fc|``.
    free
        True where the reflection is in the R-free test set.

    Returns
    -------
    RFactors
        The three R factors.

    Raises
    ------
    ValueError
        If the arrays differ in length or the working set is empty.
    """
    f_obs = np.asarray(f_obs, dtype=np.float64)
    f_calc = np.asarray(f_calc, dtype=np.float64)
    free = np.asarray(free, dtype=bool)
    if not f_obs.shape == f_calc.shape == free.shape:
        raise ValueError(
            f"f_obs {f_obs.shape}, f_calc {f_calc.shape} and free {free.shape} must be "
            "indexed by the same reflections"
        )
    if free.all():
        raise ValueError("Every reflection is flagged free; the working set is empty")

    residual = np.abs(f_obs - f_calc)

    def r_over(selection: np.ndarray) -> float:
        return float(residual[selection].sum() / f_obs[selection].sum())

    return RFactors(
        work=r_over(~free),
        free=r_over(free) if free.any() else None,
        all=float(residual.sum() / f_obs.sum()),
    )
