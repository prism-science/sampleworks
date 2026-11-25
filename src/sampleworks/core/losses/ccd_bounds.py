"""Chemical Component Dictionary (CCD) distance-bound violation penalty.

The covalent-geometry term of DEER-guided Boltz-2 (Kao, Meiler & Mchaourab, bioRxiv
2026.10.29.762916, Methods, "MolProbity surrogate and CCD bound violation"): a linear
flat-bottom penalty on every intra-residue heavy-atom distance that leaves its RDKit
distance-bounds window, plus peptide-junction terms across consecutive residues. It is a
data-free prior on chemistry, meant to sit in a
:class:`~sampleworks.core.losses.composite.CompositeLoss` next to a data term.

The bounds follow Boltz-2's ``compute_geometry_constraints`` (``boltz/data/parse/schema.py``)
and are widened exactly as its ``PoseBustersPotential`` widens them
(``boltz/model/potentials/potentials.py``): bonds and angles (1-3 pairs) by a relative buffer,
other pairs keep only a softened lower bound, and van der Waals cutoffs cap bond lengths and
floor non-bonded distances. Boltz-2 builds these bounds only for ligands, which is why the paper
computes them per standard residue; this module does the same for any residue the CCD knows.

RDKit's bounds come from UFF, whose ideal lengths sit a few hundredths of an Angstrom from
Engh-Huber (CA-C 1.486 vs 1.525 A). With the default 12.5 % buffers that offset is irrelevant;
tighten the buffers far below that and the penalty starts pulling bonds away from the values
validation tools score against.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import biotite.structure as struc
import biotite.structure.info as info
import numpy as np
import torch
from biotite.interface.rdkit import to_mol
from jaxtyping import Bool, Float, Int
from loguru import logger
from rdkit import Chem
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
from torch import Tensor


if TYPE_CHECKING:
    from biotite.structure import AtomArray

    from sampleworks.core.losses.options import CCDBoundsOptions
    from sampleworks.core.losses.protocol import LossInputs
    from sampleworks.core.losses.registry import LossBuildContext


# RDKit's own half-widths for a bond and a 1-3 pair; the peptide-junction terms reuse them so
# that they are widened by the same buffers as the intra-residue terms.
_BOND_HALF_WIDTH = 0.01
_ANGLE_HALF_WIDTH = 0.04

# (atom in residue i, atom in residue i + 1, ideal distance in A, is_bond) from the paper's
# Methods: the C-N peptide bond and the three 1-3 distances it closes.
_PEPTIDE_JUNCTION_TERMS = (
    ("C", "N", 1.329, True),
    ("CA", "N", 2.435, False),
    ("C", "CA", 2.435, False),
    ("O", "N", 2.250, False),
)

# Added to the mean van der Waals radius of a pair to give the distance no bond may exceed
# and no non-bonded pair may undercut (Boltz-2 PoseBustersPotential).
_VDW_CUTOFF_PADDING = 0.35

# Atoms equivalent by resonance or ring flip that the CCD's bond orders tell apart; structures
# name them by convention, so their bounds are merged to accept either naming.
_EQUIVALENT_ATOMS: dict[str, tuple[tuple[str, str], ...]] = {
    "ARG": (("NH1", "NH2"),),
    "ASP": (("OD1", "OD2"),),
    "GLU": (("OE1", "OE2"),),
    "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
    "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
}


@dataclass(frozen=True)
class DistanceBounds:
    """Atom-pair distance windows, before buffering.

    Attributes
    ----------
    pairs
        Atom indices of each constrained pair, shape [n_pairs, 2].
    lower, upper
        Distance bounds in Angstrom, shape [n_pairs].
    is_bond, is_angle
        Whether the pair is bonded, or the two ends of a bond angle (a 1-3 pair), shape [n_pairs].
        Three-membered rings make a pair both.
    """

    pairs: Int[np.ndarray, "n_pairs 2"]
    lower: Float[np.ndarray, " n_pairs"]
    upper: Float[np.ndarray, " n_pairs"]
    is_bond: Bool[np.ndarray, " n_pairs"]
    is_angle: Bool[np.ndarray, " n_pairs"]

    @classmethod
    def concatenate(cls, parts: list[DistanceBounds]) -> DistanceBounds:
        """Join bounds whose pair indices already refer to one atom array.

        Parameters
        ----------
        parts
            Bounds to join, in order.

        Returns
        -------
        DistanceBounds
            All pairs of ``parts``; empty when ``parts`` is.
        """
        if not parts:
            return cls(
                pairs=np.zeros((0, 2), dtype=np.int64),
                lower=np.zeros(0),
                upper=np.zeros(0),
                is_bond=np.zeros(0, dtype=bool),
                is_angle=np.zeros(0, dtype=bool),
            )
        return cls(
            pairs=np.concatenate([part.pairs for part in parts]),
            lower=np.concatenate([part.lower for part in parts]),
            upper=np.concatenate([part.upper for part in parts]),
            is_bond=np.concatenate([part.is_bond for part in parts]),
            is_angle=np.concatenate([part.is_angle for part in parts]),
        )


@functools.cache
def component_bounds(res_name: str) -> tuple[tuple[str, ...], DistanceBounds] | None:
    """RDKit distance bounds between the heavy atoms of one CCD component.

    Bounds are computed on the full component, hydrogens included, as Boltz-2 does; only the
    heavy-atom pairs are kept because the generative models do not place hydrogens.

    Parameters
    ----------
    res_name
        CCD component code, e.g. ``"ALA"``.

    Returns
    -------
    tuple[tuple[str, ...], DistanceBounds] | None
        Heavy-atom names, and bounds whose pair indices refer to that name tuple; ``None`` if the
        CCD lacks the component, RDKit cannot sanitize it, or it has fewer than two heavy atoms.
    """
    try:
        component = info.residue(res_name)
        mol = to_mol(component, explicit_hydrogen=True)
        Chem.SanitizeMol(mol)
    except (KeyError, Chem.rdchem.MolSanitizeException) as error:
        logger.warning(f"No CCD distance bounds for residue {res_name!r}: {error}")
        return None

    heavy = np.flatnonzero(component.element != "H")
    if len(heavy) < 2:
        return None

    Chem.GetSymmSSSR(mol)
    # Upper triangle holds upper bounds, lower triangle lower bounds.
    bounds = GetMoleculeBoundsMatrix(
        mol,
        set15bounds=True,
        scaleVDW=True,
        doTriangleSmoothing=True,
        useMacrocycle14config=False,
    )
    bonded = {tuple(sorted(match)) for match in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*"))}
    angle_ends = {
        tuple(sorted((match[0], match[2])))
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*~*"))
    }

    names = tuple(str(name) for name in component.atom_name[heavy])
    first, second = np.triu_indices(len(heavy), k=1)
    atom_i, atom_j = heavy[first], heavy[second]  # atom_i < atom_j in component order
    pair_keys = list(zip(atom_i.tolist(), atom_j.tolist(), strict=True))
    lower, upper = bounds[atom_j, atom_i], bounds[atom_i, atom_j]
    _merge_equivalent_bounds(res_name, names, first, second, lower, upper)
    return names, DistanceBounds(
        pairs=np.stack([first, second], axis=-1).astype(np.int64),
        lower=lower,
        upper=upper,
        is_bond=np.array([key in bonded for key in pair_keys], dtype=bool),
        is_angle=np.array([key in angle_ends for key in pair_keys], dtype=bool),
    )


def _merge_equivalent_bounds(
    res_name: str,
    names: tuple[str, ...],
    first: Int[np.ndarray, " n_pairs"],
    second: Int[np.ndarray, " n_pairs"],
    lower: Float[np.ndarray, " n_pairs"],
    upper: Float[np.ndarray, " n_pairs"],
) -> None:
    """Widen each pair's window to cover its image under swapping equivalent atoms.

    Modifies ``lower`` and ``upper`` in place.
    """
    position = {name: index for index, name in enumerate(names)}
    pair_index = {
        frozenset((a, b)): k for k, (a, b) in enumerate(zip(first.tolist(), second.tolist()))
    }
    for name_a, name_b in _EQUIVALENT_ATOMS.get(res_name, ()):
        if name_a not in position or name_b not in position:
            continue
        a, b = position[name_a], position[name_b]
        swap = {a: b, b: a}
        for key, k in pair_index.items():
            image = pair_index[frozenset(swap.get(atom, atom) for atom in key)]
            lower[k] = lower[image] = min(lower[k], lower[image])
            upper[k] = upper[image] = max(upper[k], upper[image])


def _residue_bounds(atom_array: AtomArray, start: int, stop: int) -> DistanceBounds | None:
    """Bounds for the atoms of one residue present in ``atom_array[start:stop]``."""
    template = component_bounds(str(atom_array.res_name[start]))
    if template is None:
        return None
    template_names, template_bounds = template

    index_in_array = {
        str(name): start + offset for offset, name in enumerate(atom_array.atom_name[start:stop])
    }
    # Template atom index -> atom array index, -1 for template atoms the model does not have
    # (OXT inside a chain, say).
    lookup = np.array([index_in_array.get(name, -1) for name in template_names])
    pairs = lookup[template_bounds.pairs]
    present = (pairs >= 0).all(axis=-1)
    return DistanceBounds(
        pairs=pairs[present],
        lower=template_bounds.lower[present],
        upper=template_bounds.upper[present],
        is_bond=template_bounds.is_bond[present],
        is_angle=template_bounds.is_angle[present],
    )


def _peptide_junction_bounds(
    atom_array: AtomArray, residue_starts: Int[np.ndarray, " n_residues_plus_one"]
) -> DistanceBounds:
    """Bounds across the peptide bond of every consecutive residue pair in a chain."""
    pairs, ideal, is_bond = [], [], []
    for previous, current, stop in zip(
        residue_starts[:-2], residue_starts[1:-1], residue_starts[2:], strict=True
    ):
        same_chain = atom_array.chain_id[previous] == atom_array.chain_id[current]
        adjacent = atom_array.res_id[current] - atom_array.res_id[previous] == 1
        if not (same_chain and adjacent):
            continue
        before = {
            str(name): previous + k for k, name in enumerate(atom_array.atom_name[previous:current])
        }
        after = {
            str(name): current + k for k, name in enumerate(atom_array.atom_name[current:stop])
        }
        if "C" not in before or "N" not in after:
            continue
        for name_before, name_after, distance, bond in _PEPTIDE_JUNCTION_TERMS:
            if name_before in before and name_after in after:
                pairs.append((before[name_before], after[name_after]))
                ideal.append(distance)
                is_bond.append(bond)

    if not pairs:
        return DistanceBounds.concatenate([])
    ideal_array = np.array(ideal)
    is_bond_array = np.array(is_bond, dtype=bool)
    half_width = np.where(is_bond_array, _BOND_HALF_WIDTH, _ANGLE_HALF_WIDTH)
    return DistanceBounds(
        pairs=np.array(pairs, dtype=np.int64),
        lower=ideal_array - half_width,
        upper=ideal_array + half_width,
        is_bond=is_bond_array,
        is_angle=~is_bond_array,
    )


def ccd_distance_bounds(atom_array: AtomArray) -> DistanceBounds:
    """Unbuffered CCD distance bounds for every residue and peptide junction of ``atom_array``.

    Parameters
    ----------
    atom_array
        Topology the scored coordinates follow (the model atom array).

    Returns
    -------
    DistanceBounds
        Pair indices into ``atom_array``. Residues the CCD does not describe, and single-atom
        residues such as waters and ions, contribute no pairs.
    """
    residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    per_residue = [
        _residue_bounds(atom_array, start, stop)
        for start, stop in zip(residue_starts[:-1], residue_starts[1:], strict=True)
    ]
    return DistanceBounds.concatenate(
        [bounds for bounds in per_residue if bounds is not None]
        + [_peptide_junction_bounds(atom_array, residue_starts)]
    )


def buffered_bounds(
    bounds: DistanceBounds,
    elements: np.ndarray,
    *,
    bond_buffer: float,
    angle_buffer: float,
    clash_buffer: float,
) -> tuple[Float[np.ndarray, " n_pairs"], Float[np.ndarray, " n_pairs"]]:
    """Widen distance bounds the way Boltz-2's ``PoseBustersPotential`` does.

    Parameters
    ----------
    bounds
        Unbuffered bounds.
    elements
        Element symbol of every atom the pair indices refer to.
    bond_buffer, angle_buffer
        Relative widening of bond and 1-3 bounds on both sides; a pair that is both takes the
        smaller.
    clash_buffer
        Relative lowering of the lower bound of every other pair, whose upper bound is dropped.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower and upper bounds in Angstrom, shape [n_pairs] each; upper bounds may be ``inf``.
    """
    lower, upper = bounds.lower.copy(), bounds.upper.copy()
    bond_only = bounds.is_bond & ~bounds.is_angle
    angle_only = bounds.is_angle & ~bounds.is_bond
    both = bounds.is_bond & bounds.is_angle
    neither = ~bounds.is_bond & ~bounds.is_angle
    for mask, buffer in (
        (bond_only, bond_buffer),
        (angle_only, angle_buffer),
        (both, min(bond_buffer, angle_buffer)),
    ):
        lower[mask] *= 1.0 - buffer
        upper[mask] *= 1.0 + buffer
    lower[neither] *= 1.0 - clash_buffer
    upper[neither] = np.inf

    vdw_radii = np.array([info.vdw_radius_single(element) or 0.0 for element in elements])
    cutoffs = _VDW_CUTOFF_PADDING + vdw_radii[bounds.pairs].mean(axis=-1)
    lower[~bounds.is_bond] = np.maximum(lower[~bounds.is_bond], cutoffs[~bounds.is_bond])
    upper[bounds.is_bond] = np.minimum(upper[bounds.is_bond], cutoffs[bounds.is_bond])
    return lower, upper


class CCDBoundsLossFunction:
    """Flat-bottom penalty on atom-pair distances outside their CCD bounds.

    Each pair contributes how far its distance falls below its lower bound or above its upper
    bound, in Angstrom; the loss sums that over pairs and batch members. The topology is only
    known once the model atom array exists, so the bounds are built in :meth:`prepare`.

    Parameters
    ----------
    bond_buffer
        Relative widening of bond-length bounds.
    angle_buffer
        Relative widening of 1-3 distance bounds.
    clash_buffer
        Relative lowering of the lower bound of all other intra-residue pairs.
    """

    def __init__(
        self, bond_buffer: float = 0.125, angle_buffer: float = 0.125, clash_buffer: float = 0.10
    ):
        self.bond_buffer = bond_buffer
        self.angle_buffer = angle_buffer
        self.clash_buffer = clash_buffer
        self._pairs: Int[Tensor, "n_pairs 2"] | None = None
        self._lower: Float[Tensor, " n_pairs"] | None = None
        self._upper: Float[Tensor, " n_pairs"] | None = None

    def prepare(self, loss_inputs: LossInputs, *, device: torch.device | str = "cpu") -> None:
        """Build the distance bounds for the model atom array the coordinates follow.

        Parameters
        ----------
        loss_inputs
            Model-order loss inputs; their ``atom_array`` supplies the topology.
        device
            PyTorch device the bounds are placed on.

        Raises
        ------
        ValueError
            If ``loss_inputs`` carries no atom array.
        """
        atom_array = loss_inputs.atom_array
        if atom_array is None:
            raise ValueError("ccd_bounds needs LossInputs built with an atom array.")
        bounds = ccd_distance_bounds(atom_array)
        lower, upper = buffered_bounds(
            bounds,
            atom_array.element,
            bond_buffer=self.bond_buffer,
            angle_buffer=self.angle_buffer,
            clash_buffer=self.clash_buffer,
        )
        logger.info(
            f"ccd_bounds: {len(bounds.pairs)} pairs ({int(bounds.is_bond.sum())} bonds, "
            f"{int(bounds.is_angle.sum())} 1-3 pairs) over {atom_array.array_length()} atoms"
        )
        self._pairs = torch.as_tensor(bounds.pairs, dtype=torch.long, device=device)
        self._lower = torch.as_tensor(lower, dtype=torch.float32, device=device)
        self._upper = torch.as_tensor(upper, dtype=torch.float32, device=device)

    def __call__(
        self,
        coordinates: Float[Tensor, "batch n_atoms 3"],
        elements: Int[Tensor, "batch n_atoms"],
        b_factors: Float[Tensor, "batch n_atoms"],
        occupancies: Float[Tensor, "batch n_atoms"],
    ) -> Float[Tensor, ""]:
        """Total bound violation of ``coordinates``; the other inputs are unused.

        Parameters
        ----------
        coordinates
            Atomic coordinates in model atom order, shape [batch, n_atoms, 3].
        elements, b_factors, occupancies
            Accepted for ``LossFunctionProtocol`` compatibility and ignored.

        Returns
        -------
        Float[Tensor, ""]
            Summed violation in Angstrom.

        Raises
        ------
        RuntimeError
            If called before :meth:`prepare`.
        """
        if self._pairs is None or self._lower is None or self._upper is None:
            raise RuntimeError("Call prepare() before scoring with ccd_bounds.")
        distances = (
            coordinates[..., self._pairs[:, 0], :] - coordinates[..., self._pairs[:, 1], :]
        ).norm(dim=-1)  # [batch, n_pairs]
        too_short = (self._lower - distances).clamp(min=0)
        # An infinite upper bound gives -inf here, which the clamp turns into 0.
        too_long = (distances - self._upper).clamp(min=0)
        return (too_short + too_long).sum()


def build_ccd_bounds_loss(
    options: CCDBoundsOptions, context: LossBuildContext
) -> CCDBoundsLossFunction:
    """Build the CCD bound-violation penalty from its configured options.

    Parameters
    ----------
    options
        See :class:`~sampleworks.core.losses.options.CCDBoundsOptions`.
    context
        Run-level inputs. Unused: the topology and device arrive in ``prepare()``.

    Returns
    -------
    CCDBoundsLossFunction
        Unprepared penalty.
    """
    del context
    return CCDBoundsLossFunction(
        bond_buffer=options.bond_buffer,
        angle_buffer=options.angle_buffer,
        clash_buffer=options.clash_buffer,
    )
