"""Score an ensemble's R factors against a structure-factor target, with a selectable engine.

The R factor itself is :func:`sampleworks.eval.metrics.r_factors`. This module produces what
it compares: observed amplitudes ``|Fo|``, model amplitudes ``|Fc|`` and the R-free flags, on
one reflection list. Two engines compute ``|Fc|``:

``sfc``
    SFcalculator, through ``StructureFactorLossFunction``: the same model the SF loss
    guides with. Reflections are SFcalculator's observed set minus its outliers.
``lunus``
    ``lunus.sf``, through :mod:`sampleworks.core.forward_models.xray.lunus_sf`: the
    same model the ``diffuse_bragg`` loss guides with. Reflections are the MTZ's rows with
    a finite amplitude inside the resolution cutoff.

Arms guided by different losses are only comparable when one engine scores them all.

The ensemble is scored the way both losses score it: each conformer's ``F`` is summed in the
complex plane at weight ``1/E``, with the B-factors of the first model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import reciprocalspaceship as rs
import torch
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArrayStack, stack as stack_models
from jaxtyping import Bool, Float
from loguru import logger

from sampleworks.core.losses.protocol import LossInputs
from sampleworks.eval.metrics import r_factors, RFactors
from sampleworks.synthetic.synthetic_utils import resolve_mtz_column
from sampleworks.utils.atom_array_utils import load_structure_with_altlocs
from sampleworks.utils.atom_reconciler import AtomReconciler
from sampleworks.utils.frame_transforms import apply_forward_transform


Engine = Literal["sfc", "lunus"]
ENGINES: tuple[Engine, ...] = ("sfc", "lunus")


@dataclass(frozen=True)
class StructureFactorTarget:
    """What an ensemble is scored against.

    Attributes
    ----------
    mtz
        Target MTZ.
    resolution
        High-resolution cutoff in Angstrom.
    amplitude_column, sigma_column
        Observed amplitude column and its sigma (SFcalculator requires one). ``None``
        takes the MTZ's sole column of that type.
    free_flag_column
        R-free flag column; ``None`` takes the MTZ's sole integer column. Must exist: a
        missing column would otherwise make every reflection working set.
    test_set_value
        Flag value marking the test set: 1 is the Phenix convention the synthetic generators
        write, 0 the CCP4 one.
    bulk_solvent
        ``"off"`` scores ``|Fprotein|``; ``"combined"`` (SFC only) or ``"per_conformer"``
        adds default-scaled bulk solvent, as the loss of that name does.
    """

    mtz: Path
    resolution: float
    amplitude_column: str | None = None
    sigma_column: str | None = None
    free_flag_column: str | None = None
    test_set_value: int = 1
    bulk_solvent: str = "off"


@dataclass(frozen=True)
class ReflectionAmplitudes:
    """Observed and model amplitudes on one reflection list.

    Attributes
    ----------
    f_obs, f_calc
        ``|Fo|`` and ``|Fc|``, ``[n_refl]``.
    free
        True where the reflection is in the R-free test set, ``[n_refl]``.
    """

    f_obs: Float[np.ndarray, " n_refl"]
    f_calc: Float[np.ndarray, " n_refl"]
    free: Bool[np.ndarray, " n_refl"]


def load_ensemble(
    path: Path, frame_reference: Path | None = None
) -> tuple[AtomArrayStack, Float[np.ndarray, "n_models n_atoms 3"]]:
    """Load a (multi-model) ensemble, optionally superposed onto a crystal-frame reference.

    Parameters
    ----------
    path
        Ensemble file; each model is one conformer.
    frame_reference
        Structure already in the target's crystal frame. When given, one rigid transform,
        fitted from the ensemble's mean coordinates to the reference on their common atoms,
        moves every conformer, so the ensemble's internal geometry is untouched. ``None``
        scores the coordinates as they are.

    Returns
    -------
    tuple of (AtomArrayStack, numpy.ndarray)
        The ensemble, and its coordinates ``[n_models, n_atoms, 3]`` in the scoring frame.
    """
    # Without bonds: inferring them drops atoms atomworks takes for leaving groups (a
    # C-terminal OXT among them), and every atom in the file scatters.
    loaded = load_any(path, extra_fields=["b_factor", "occupancy"], include_bonds=False)
    stack = loaded if isinstance(loaded, AtomArrayStack) else stack_models([loaded])
    coords = np.asarray(stack.coord, dtype=np.float32)
    if frame_reference is None:
        return stack, coords

    reference = load_structure_with_altlocs(frame_reference)
    reference = reference[0] if isinstance(reference, AtomArrayStack) else reference
    reconciler = AtomReconciler.from_arrays(stack[0], reference)
    mean = torch.from_numpy(coords.mean(axis=0, keepdims=True))  # [1, n_atoms, 3]
    reference_coords = torch.as_tensor(reference.coord, dtype=torch.float32)[None]
    _, transform = reconciler.align(mean, reconciler.struct_to_model(reference_coords, mean))
    n_models = coords.shape[0]
    per_model = {
        "rotation": transform["rotation"].expand(n_models, 3, 3),
        "translation": transform["translation"].expand(n_models, 3),
    }
    aligned = apply_forward_transform(torch.from_numpy(coords), per_model)
    logger.info(
        f"Superposed {path.name} onto {frame_reference.name} on {reconciler.n_common} common "
        f"atoms (translation {transform['translation'][0].tolist()})"
    )
    return stack, aligned.numpy()


def ensemble_loss_inputs(
    stack: AtomArrayStack, coords: np.ndarray, device: torch.device
) -> LossInputs:
    """Build ``LossInputs`` for an ensemble: one topology, per-conformer coordinates.

    Parameters
    ----------
    stack
        The ensemble; its first model gives the topology, B-factors and elements.
    coords
        Coordinates ``[n_models, n_atoms, 3]``.
    device
        Device for the tensors.

    Returns
    -------
    LossInputs
        Occupancies ``1/n_models``, so a complex sum over conformers is the ensemble total.
    """
    template = LossInputs.from_atom_array(stack[0], ensemble_size=len(coords), device=device)
    return LossInputs(
        elements=template.elements,
        b_factors=template.b_factors,
        occupancies=template.occupancies,
        input_coords=torch.from_numpy(np.ascontiguousarray(coords)).to(
            dtype=torch.float32, device=device
        ),
        atom_array=template.atom_array,
    )


def resolve_target_columns(target: StructureFactorTarget) -> StructureFactorTarget:
    """Fill in a target's ``None`` columns from its MTZ, failing on any ambiguity.

    Parameters
    ----------
    target
        Target with possibly unresolved columns.

    Returns
    -------
    StructureFactorTarget
        The same target with every column named.
    """
    dataset = rs.read_mtz(str(target.mtz))
    return StructureFactorTarget(
        mtz=target.mtz,
        resolution=target.resolution,
        amplitude_column=resolve_mtz_column(
            dataset, rs.StructureFactorAmplitudeDtype(), column=target.amplitude_column
        ),
        sigma_column=resolve_mtz_column(
            dataset, rs.StandardDeviationDtype(), column=target.sigma_column
        ),
        free_flag_column=resolve_mtz_column(
            dataset, rs.MTZIntDtype(), column=target.free_flag_column
        ),
        test_set_value=target.test_set_value,
        bulk_solvent=target.bulk_solvent,
    )


def sfc_amplitudes(
    loss_inputs: LossInputs,
    target: StructureFactorTarget,
    *,
    batch_partition: int = 4,
    device: torch.device,
) -> ReflectionAmplitudes:
    """Model amplitudes from SFcalculator, on its observed reflections minus outliers.

    Parameters
    ----------
    loss_inputs
        The ensemble, from :func:`ensemble_loss_inputs`.
    target
        Target with every column resolved.
    batch_partition
        Conformers per SFcalculator chunk; lower it on a CUDA OOM.
    device
        Device to compute on.

    Returns
    -------
    ReflectionAmplitudes
        ``|Fo|``, ``|Fc|`` and free flags.
    """
    from sampleworks.core.losses.structure_factor import StructureFactorLossFunction

    loss_fn = StructureFactorLossFunction(
        target.mtz,
        expcolumns=[str(target.amplitude_column), str(target.sigma_column)],
        resolution=target.resolution,
        bulk_solvent=target.bulk_solvent,
        exclude_free_reflections=False,
        normalize_amplitude=False,
        batch_partition=batch_partition,
        sfcalculator_kwargs={
            "freeflag": target.free_flag_column,
            "testset_value": target.test_set_value,
        },
    )
    loss_fn.prepare(loss_inputs, device=device)
    sfc = loss_fn.sfc
    assert sfc is not None  # prepare() built it
    with torch.no_grad():
        f_calc = loss_fn.calc_fmodel(
            loss_inputs.input_coords, loss_inputs.b_factors, loss_inputs.occupancies
        ).abs()
    kept = ~sfc.Outlier
    return ReflectionAmplitudes(
        f_obs=sfc.Fo[kept].cpu().numpy(),
        f_calc=f_calc[kept].cpu().numpy(),
        free=np.asarray(sfc.free_flag[kept], dtype=bool),
    )


def lunus_amplitudes(
    loss_inputs: LossInputs, target: StructureFactorTarget, *, device: torch.device
) -> ReflectionAmplitudes:
    """Model amplitudes from lunus.sf, on the MTZ rows with a finite amplitude.

    Builds the setup and, for ``bulk_solvent="per_conformer"``, the calibrated solvent model
    exactly as ``DiffuseBraggLossFunction.prepare`` does.

    Parameters
    ----------
    loss_inputs
        The ensemble, from :func:`ensemble_loss_inputs`.
    target
        Target with every column resolved; ``bulk_solvent`` ``"off"`` or ``"per_conformer"``.
    device
        Device to compute on.

    Returns
    -------
    ReflectionAmplitudes
        ``|Fo|``, ``|<F>|`` over the conformers, and free flags.

    Raises
    ------
    ValueError
        For ``bulk_solvent="combined"``, which lunus has no model for.
    """
    from sampleworks.core.forward_models.xray import lunus_sf

    if target.bulk_solvent not in ("off", "per_conformer"):
        raise ValueError(
            f"lunus scores bulk_solvent 'off' or 'per_conformer', not {target.bulk_solvent!r}"
        )
    dataset = rs.read_mtz(str(target.mtz))
    dataset.compute_dHKL(inplace=True)
    f_obs = dataset[target.amplitude_column].to_numpy(dtype=np.float64)
    flags = dataset[target.free_flag_column].to_numpy(dtype=np.float64, na_value=np.nan)
    kept = np.isfinite(f_obs) & (dataset["dHKL"].to_numpy() >= target.resolution)
    hkl = dataset.get_hkls()[kept]

    # to_atom_array carries the B-factors the setup bakes into its kernels.
    atom_array = loss_inputs.to_atom_array()
    setup = lunus_sf.build_setup(
        atom_array, dataset.cell, dataset.spacegroup.hm, target.resolution, device=device
    )
    solvent = None
    if target.bulk_solvent == "per_conformer":
        solvent_fraction = lunus_sf.geometric_solvent_fraction(
            atom_array, atom_array.coord, dataset.cell, dataset.spacegroup, setup.grid_shape
        )
        solvent = lunus_sf.calibrate_solvent(
            setup,
            torch.as_tensor(atom_array.coord, dtype=torch.float32, device=device),
            torch.ones(atom_array.array_length(), device=device),
            solvent_fraction,
        )
    with torch.no_grad():
        # Full occupancy per conformer, then the mean: the 1/E-weighted complex sum.
        f_configs = lunus_sf.structure_factors(
            setup,
            loss_inputs.input_coords,
            loss_inputs.occupancies * loss_inputs.input_coords.shape[0],
            torch.as_tensor(hkl, dtype=torch.long, device=device),
            solvent=solvent,
        )  # [n_configs, n_refl]
        f_calc = f_configs.mean(dim=0).abs()
    return ReflectionAmplitudes(
        f_obs=f_obs[kept],
        f_calc=f_calc.cpu().numpy(),
        free=flags[kept] == target.test_set_value,
    )


def ensemble_r_factors(
    loss_inputs: LossInputs,
    target: StructureFactorTarget,
    *,
    engine: Engine = "sfc",
    batch_partition: int = 4,
    device: torch.device,
) -> RFactors:
    """R factors of an ensemble already in ``LossInputs`` form, e.g. a run's model atom space.

    Parameters
    ----------
    loss_inputs
        The ensemble: ``input_coords`` ``[n_conformers, n_atoms, 3]``, an ``atom_array`` for
        the topology, and occupancies ``1/n_conformers``.
    target
        What to score against; ``None`` columns are resolved from the MTZ.
    engine
        ``"sfc"`` or ``"lunus"``; see the module docstring.
    batch_partition
        SFC only: conformers per chunk.
    device
        Device to compute on.

    Returns
    -------
    RFactors
        R-work, R-free and R-all.
    """
    target = resolve_target_columns(target)
    if engine == "sfc":
        amplitudes = sfc_amplitudes(
            loss_inputs, target, batch_partition=batch_partition, device=device
        )
    elif engine == "lunus":
        amplitudes = lunus_amplitudes(loss_inputs, target, device=device)
    else:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
    return r_factors(amplitudes.f_obs, amplitudes.f_calc, amplitudes.free)


def score_ensemble(
    path: Path,
    target: StructureFactorTarget,
    *,
    engine: Engine = "sfc",
    frame_reference: Path | None = None,
    batch_partition: int = 4,
    device: torch.device,
) -> tuple[RFactors, dict]:
    """R factors of one ensemble file against a structure-factor target.

    Parameters
    ----------
    path
        Ensemble file (e.g. a run's ``refined.cif``).
    target
        What to score against; ``None`` columns are resolved from the MTZ.
    engine
        ``"sfc"`` or ``"lunus"``; see the module docstring.
    frame_reference
        Crystal-frame structure to superpose the ensemble onto first; see
        :func:`load_ensemble`.
    batch_partition
        SFC only: conformers per chunk.
    device
        Device to compute on.

    Returns
    -------
    tuple of (RFactors, dict)
        The R factors, and a JSON-ready record of them with everything that produced them.
    """
    target = resolve_target_columns(target)
    stack, coords = load_ensemble(path, frame_reference)
    loss_inputs = ensemble_loss_inputs(stack, coords, device)
    r = ensemble_r_factors(
        loss_inputs, target, engine=engine, batch_partition=batch_partition, device=device
    )
    record = {
        "structure": str(path),
        "mtz": str(target.mtz),
        "resolution": target.resolution,
        "engine": engine,
        "amplitude_column": target.amplitude_column,
        "free_flag_column": target.free_flag_column,
        "test_set_value": target.test_set_value,
        "bulk_solvent": target.bulk_solvent,
        "frame_reference": None if frame_reference is None else str(frame_reference),
        "n_conformers": int(coords.shape[0]),
        "n_atoms": int(coords.shape[1]),
        "r_work": r.work,
        "r_free": r.free,
        "r_all": r.all,
    }
    return r, record
