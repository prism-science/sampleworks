"""Differentiable heavy-atom surrogate of the MolProbity score.

The structure-quality term of DEER-guided Boltz-2 (Kao, Meiler & Mchaourab, bioRxiv
2026.10.29.762916, Methods, "MolProbity surrogate and CCD bound violation"). It keeps the form
and weights of the MolProbity score (Chen et al., Acta Cryst. D 66, 12-21, 2010) without its
constant,

    0.426 ln(1 + c) + 0.33 ln(1 + max(0, r - 1)) + 0.25 ln(1 + max(0, a - 2)),

with soft versions of its three inputs:

- ``c``, the clashscore: a logistic count, per 1000 heavy atoms, of heavy-atom pairs more than
  one residue apart whose van der Waals overlap exceeds 0.40 A (CCP4 radii, which are Bondi's;
  logistic width 0.10 A).
- ``r``, the percentage of side chains whose rotamer lies below 0.3 % in the Top8000
  distributions.
- ``a``, the percentage of residues outside the Top8000 Ramachandran favored regions (below 2 %).

The Top8000 percentile-contour grids (Williams et al., Protein Sci. 27, 293-315, 2018; CC BY 4.0,
https://github.com/rlabduke/reference_data) are packed by ``scripts/build_molprobity_tables.py``
and interpolated multilinearly, as MolProbity does. The paper does not say how it softens the
rotamer and Ramachandran indicators; here each is a logistic in log contour value, so its
gradient reaches residues near a contour but not those deep inside an empty region, where the
grids are flat zero.
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING

import biotite.structure as struc
import biotite.structure.info as info
import numpy as np
import torch
from jaxtyping import Float, Int
from loguru import logger
from torch import Tensor


if TYPE_CHECKING:
    from biotite.structure import AtomArray

    from sampleworks.core.losses.options import MolProbityOptions
    from sampleworks.core.losses.protocol import LossInputs
    from sampleworks.core.losses.registry import LossBuildContext


CLASH_OVERLAP = 0.40
CLASH_WIDTH = 0.10
ROTAMER_OUTLIER = 0.003
RAMACHANDRAN_FAVORED = 0.02

# Atom quadruples of each side-chain dihedral, keyed by the rotamer table that scores them.
_CHI_ATOMS: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "ARG": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"),
            ("CG", "CD", "NE", "CZ")),
    "ASN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "ASP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "CYS": (("N", "CA", "CB", "SG"),),
    "GLN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")),
    "GLU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")),
    "HIS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
    "ILE": (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")),
    "LEU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "LYS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"),
            ("CG", "CD", "CE", "NZ")),
    "MET": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")),
    "PHE_TYR": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "PRO": (("N", "CA", "CB", "CG"),),
    "SER": (("N", "CA", "CB", "OG"),),
    "THR": (("N", "CA", "CB", "OG1"),),
    "TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "VAL": (("N", "CA", "CB", "CG1"),),
}  # fmt: skip

_ROTAMER_TABLE = {name: name for name in _CHI_ATOMS} | {
    "PHE": "PHE_TYR",
    "TYR": "PHE_TYR",
    "MSE": "MET",
}

_RAMACHANDRAN_TABLES = (
    "general",
    "glycine",
    "cis_proline",
    "trans_proline",
    "pre_proline",
    "isoleucine_valine",
)
_GENERAL, _GLYCINE, _CIS_PROLINE, _TRANS_PROLINE, _PRE_PROLINE, _ILE_VAL = range(6)

_SCORE_WEIGHTS = (0.426, 0.33, 0.25)


@dataclass(frozen=True)
class ContourGrid:
    """A periodic percentile-contour grid over dihedral angles.

    Attributes
    ----------
    grid
        Contour values at bin centers, shape [n_bins_1, ..., n_bins_k].
    lower, width
        Lower bound and bin width of each axis in degrees, shape [k].
    """

    grid: Tensor
    lower: Tensor
    width: Tensor

    def to(self, device: torch.device | str) -> ContourGrid:
        """Return this grid on ``device``."""
        return ContourGrid(self.grid.to(device), self.lower.to(device), self.width.to(device))

    def __call__(self, angles: Float[Tensor, "*batch k"]) -> Float[Tensor, "*batch"]:
        """Interpolate the grid multilinearly at ``angles`` (degrees), wrapping every axis.

        Parameters
        ----------
        angles
            Dihedral angles, last axis matching the grid's dimensions.

        Returns
        -------
        Tensor
            Contour values, shape ``angles.shape[:-1]``.
        """
        n_bins = torch.tensor(self.grid.shape, device=angles.device)
        position = (angles - self.lower) / self.width - 0.5  # in bins, from the first center
        below = torch.floor(position)
        fraction = position - below
        below = below.long()
        value = angles.new_zeros(angles.shape[:-1])
        for corner in itertools.product((0, 1), repeat=len(self.grid.shape)):
            offset = torch.tensor(corner, device=angles.device)
            index = torch.remainder(below + offset, n_bins)
            weight = torch.where(offset.bool(), fraction, 1.0 - fraction).prod(dim=-1)
            value = value + weight * self.grid[tuple(index.unbind(dim=-1))]
        return value


@functools.cache
def _contour_grids() -> dict[tuple[str, str], ContourGrid]:
    """Load the packed Top8000 grids on the CPU.

    Returns
    -------
    dict[tuple[str, str], ContourGrid]
        Keyed ``("rama", table)`` or ``("rota", table)``; the ``.npz`` stores each array under
        ``"<kind>/<table>/<field>"``.
    """
    with files("sampleworks.data").joinpath("molprobity_top8000.npz").open("rb") as handle:
        packed = np.load(handle)
        keys = {tuple(key.split("/")[:2]) for key in packed.files if key.endswith("/grid")}
        return {
            (kind, table): ContourGrid(
                torch.from_numpy(packed[f"{kind}/{table}/grid"]),
                torch.from_numpy(packed[f"{kind}/{table}/lower"]),
                torch.from_numpy(packed[f"{kind}/{table}/width"]),
            )
            for kind, table in keys
        }


def dihedral(points: Float[Tensor, "*batch 4 3"]) -> Float[Tensor, "*batch"]:
    """Dihedral angle in degrees, in (-180, 180], of four points.

    Parameters
    ----------
    points
        The four points along the second-to-last axis.

    Returns
    -------
    Tensor
        Dihedral angles in degrees.
    """
    b0 = points[..., 0, :] - points[..., 1, :]
    b1 = points[..., 2, :] - points[..., 1, :]
    b2 = points[..., 3, :] - points[..., 2, :]
    b1_unit = b1 / b1.norm(dim=-1, keepdim=True)
    v = b0 - (b0 * b1_unit).sum(-1, keepdim=True) * b1_unit
    w = b2 - (b2 * b1_unit).sum(-1, keepdim=True) * b1_unit
    x = (v * w).sum(-1)
    y = (torch.cross(b1_unit, v, dim=-1) * w).sum(-1)
    return torch.rad2deg(torch.atan2(y, x))


def soft_below(
    values: Float[Tensor, "*batch"], threshold: float, softness: float
) -> Float[Tensor, "*batch"]:
    """Logistic indicator of ``values < threshold``, on a log scale.

    Parameters
    ----------
    values
        Non-negative contour values.
    threshold
        Contour level the indicator switches at.
    softness
        Width of the switch in natural-log units of the value.

    Returns
    -------
    Tensor
        Values in (0, 1), 0.5 at the threshold.
    """
    log_values = torch.log(values.clamp(min=0.0) + 1e-6)
    return torch.sigmoid((np.log(threshold) - log_values) / softness)


@functools.cache
def _amino_acids() -> frozenset[str]:
    """Residue names the CCD classifies as amino acids."""
    return frozenset(info.amino_acid_names())


def _residue_atoms(atom_array: AtomArray, start: int, stop: int) -> dict[str, int]:
    """Atom name to atom-array index for the residue spanning ``start:stop``."""
    return {str(name): start + k for k, name in enumerate(atom_array.atom_name[start:stop])}


def _is_bonded_successor(atom_array: AtomArray, previous: int, current: int) -> bool:
    """Whether the residue starting at ``current`` follows the one at ``previous`` in its chain."""
    return bool(
        atom_array.chain_id[previous] == atom_array.chain_id[current]
        and atom_array.res_id[current] - atom_array.res_id[previous] == 1
    )


@dataclass(frozen=True)
class RamachandranSites:
    """Atoms and table of every residue with a defined phi and psi.

    Attributes
    ----------
    atoms
        C(i-1), N, CA, C, N(i+1) of each residue, shape [n_residues, 5].
    previous_ca
        CA(i-1), for the omega that tells cis from trans proline, shape [n_residues].
    table
        Index into ``_RAMACHANDRAN_TABLES``; prolines hold ``_TRANS_PROLINE`` and are switched
        to cis by their omega when scored, shape [n_residues].
    """

    atoms: Int[np.ndarray, "n_residues 5"]
    previous_ca: Int[np.ndarray, " n_residues"]
    table: Int[np.ndarray, " n_residues"]


def _ramachandran_table(res_name: str, next_res_name: str) -> int:
    """Index into ``_RAMACHANDRAN_TABLES`` of a residue, by its own and the next residue's name.

    Prolines get the trans table; scoring switches them to cis by their omega.
    """
    if res_name == "GLY":
        return _GLYCINE
    if res_name == "PRO":
        return _TRANS_PROLINE
    if next_res_name == "PRO":
        return _PRE_PROLINE
    if res_name in ("ILE", "VAL"):
        return _ILE_VAL
    return _GENERAL


def ramachandran_sites(atom_array: AtomArray) -> RamachandranSites:
    """Find the residues MolProbity would put on a Ramachandran plot, and their tables.

    Parameters
    ----------
    atom_array
        Topology the scored coordinates follow.

    Returns
    -------
    RamachandranSites
        One entry per amino-acid residue with bonded neighbours on both sides.
    """
    residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    residues = [
        _residue_atoms(atom_array, start, stop)
        for start, stop in zip(residue_starts[:-1], residue_starts[1:])
    ]
    res_names = [str(atom_array.res_name[start]) for start in residue_starts[:-1]]
    atoms, previous_ca, table = [], [], []
    for k in range(1, len(residues) - 1):
        before, here, after = residues[k - 1], residues[k], residues[k + 1]
        bonded = _is_bonded_successor(
            atom_array, residue_starts[k - 1], residue_starts[k]
        ) and _is_bonded_successor(atom_array, residue_starts[k], residue_starts[k + 1])
        needed = {"C", "CA"} <= before.keys() and {"N", "CA", "C"} <= here.keys() and "N" in after
        if not (bonded and needed and res_names[k] in _amino_acids()):
            continue
        atoms.append((before["C"], here["N"], here["CA"], here["C"], after["N"]))
        previous_ca.append(before["CA"])
        table.append(_ramachandran_table(res_names[k], res_names[k + 1]))
    return RamachandranSites(
        atoms=np.array(atoms, dtype=np.int64).reshape(-1, 5),
        previous_ca=np.array(previous_ca, dtype=np.int64),
        table=np.array(table, dtype=np.int64),
    )


def rotamer_sites(atom_array: AtomArray) -> dict[str, Int[np.ndarray, "n_residues n_chi 4"]]:
    """Chi-angle atom quadruples of every side chain with a rotamer table, keyed by table.

    Parameters
    ----------
    atom_array
        Topology the scored coordinates follow.

    Returns
    -------
    dict[str, np.ndarray]
        Side chains missing any chi atom are left out.
    """
    residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    sites: dict[str, list] = {}
    for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
        table = _ROTAMER_TABLE.get(str(atom_array.res_name[start]))
        if table is None:
            continue
        residue = _residue_atoms(atom_array, start, stop)
        if str(atom_array.res_name[start]) == "MSE":
            residue["SD"] = residue.get("SE", -1)
        quadruples = [[residue.get(name, -1) for name in chi] for chi in _CHI_ATOMS[table]]
        if min(min(chi) for chi in quadruples) >= 0:
            sites.setdefault(table, []).append(quadruples)
    return {table: np.array(quadruples, dtype=np.int64) for table, quadruples in sites.items()}


@dataclass(frozen=True)
class ClashTopology:
    """Heavy atoms the clashscore counts, with what decides which of their pairs may clash.

    Attributes
    ----------
    atoms
        Atom-array indices of the counted heavy atoms (no solvent), shape [n_heavy].
    radii
        Van der Waals radius of each, in Angstrom, shape [n_heavy].
    chain, residue
        Chain index and residue number of each, shape [n_heavy]; residue numbers rather than
        positions so that a gap in the model does not count as adjacency.
    disulfide_sulfur
        Whether each is a cysteine SG, shape [n_heavy].
    polar
        Whether each is a nitrogen or oxygen, shape [n_heavy].
    """

    atoms: Int[Tensor, " n_heavy"]
    radii: Float[Tensor, " n_heavy"]
    chain: Int[Tensor, " n_heavy"]
    residue: Int[Tensor, " n_heavy"]
    disulfide_sulfur: Tensor
    polar: Tensor


def clash_topology(atom_array: AtomArray, device: torch.device | str) -> ClashTopology:
    """Collect the clashscore's heavy atoms and their pair-exclusion attributes.

    Parameters
    ----------
    atom_array
        Topology the scored coordinates follow.
    device
        PyTorch device the tensors are placed on.

    Returns
    -------
    ClashTopology
        One entry per non-solvent heavy atom.
    """
    heavy = (atom_array.element != "H") & ~struc.filter_solvent(atom_array)
    radii = np.array([info.vdw_radius_single(element) or 1.7 for element in atom_array.element])
    chain_index = np.unique(atom_array.chain_id, return_inverse=True)[1]
    disulfide_sulfur = (atom_array.res_name == "CYS") & (atom_array.atom_name == "SG")
    polar = np.isin(atom_array.element, ("N", "O"))
    return ClashTopology(
        atoms=torch.as_tensor(np.flatnonzero(heavy), device=device),
        radii=torch.as_tensor(radii[heavy], dtype=torch.float32, device=device),
        chain=torch.as_tensor(chain_index[heavy], device=device),
        residue=torch.as_tensor(atom_array.res_id[heavy], device=device),
        disulfide_sulfur=torch.as_tensor(disulfide_sulfur[heavy], device=device),
        polar=torch.as_tensor(polar[heavy], device=device),
    )


def close_pairs(
    positions: Float[Tensor, "n_atoms 3"], reach: float | Tensor, chunk_size: int
) -> tuple[Int[Tensor, " n_pairs"], Int[Tensor, " n_pairs"]]:
    """Index pairs ``i < j`` of points closer than ``reach``, searched in row chunks.

    Parameters
    ----------
    positions
        Points to search; no gradient flows through the search.
    reach
        Distance below which a pair is returned.
    chunk_size
        Rows of the distance matrix held in memory at a time.

    Returns
    -------
    tuple[Tensor, Tensor]
        First and second index of every close pair.
    """
    n_atoms = len(positions)
    first, second = [], []
    with torch.no_grad():
        for start in range(0, n_atoms, chunk_size):
            rows = torch.arange(start, min(start + chunk_size, n_atoms), device=positions.device)
            close = torch.cdist(positions[rows], positions) < reach
            close &= rows[:, None] < torch.arange(n_atoms, device=positions.device)
            row_index, column_index = close.nonzero(as_tuple=True)
            first.append(rows[row_index])
            second.append(column_index)
    return torch.cat(first), torch.cat(second)


# Rotamer tables whose last chi is symmetric under a 180 degree flip.
_HALF_TURN_LAST_CHI = frozenset({"ASP", "GLU", "PHE_TYR"})


class MolProbityLossFunction:
    """Soft heavy-atom MolProbity score of each structure, summed over the batch.

    Lower is better, like the MolProbity score itself. The topology (clash exclusions, phi/psi
    and chi atoms, Ramachandran categories) is built in :meth:`prepare`.

    Parameters
    ----------
    outlier_softness
        Width, in natural-log units of the contour value, of the logistic that softens the
        rotamer-outlier and Ramachandran-favored indicators.
    clash_chunk_size
        Rows of the pairwise-distance matrix searched for clashes at a time; bounds memory.
    skip_polar_pairs
        Leave nitrogen/oxygen pairs out of the clashscore. Without hydrogens, hydrogen bonds look
        like 0.2-0.5 A overlaps and dominate the soft count; the paper counts them, so the default
        does too.
    """

    def __init__(
        self,
        outlier_softness: float = 0.5,
        clash_chunk_size: int = 2048,
        skip_polar_pairs: bool = False,
    ):
        self.outlier_softness = outlier_softness
        self.clash_chunk_size = clash_chunk_size
        self.skip_polar_pairs = skip_polar_pairs
        self._prepared = False

    def prepare(self, loss_inputs: LossInputs, *, device: torch.device | str = "cpu") -> None:
        """Build clash exclusions, Ramachandran sites and rotamer sites from the topology.

        Parameters
        ----------
        loss_inputs
            Model-order loss inputs; their ``atom_array`` supplies the topology.
        device
            PyTorch device the topology and grids are placed on.

        Raises
        ------
        ValueError
            If ``loss_inputs`` carries no atom array.
        """
        atom_array = loss_inputs.atom_array
        if atom_array is None:
            raise ValueError("molprobity needs LossInputs built with an atom array.")

        self._clash = clash_topology(atom_array, device)
        ramachandran = ramachandran_sites(atom_array)
        self._rama_atoms = torch.as_tensor(ramachandran.atoms, device=device)
        self._rama_previous_ca = torch.as_tensor(ramachandran.previous_ca, device=device)
        self._rama_table = torch.as_tensor(ramachandran.table, device=device)
        self._rotamer_atoms = {
            table: torch.as_tensor(quadruples, device=device)
            for table, quadruples in rotamer_sites(atom_array).items()
        }
        self._grids = {name: grid.to(device) for name, grid in _contour_grids().items()}
        self._prepared = True
        logger.info(
            f"molprobity: {len(self._clash.atoms)} heavy atoms, "
            f"{len(ramachandran.table)} Ramachandran residues, "
            f"{sum(len(q) for q in self._rotamer_atoms.values())} rotamers"
        )

    def clashscore(self, coordinates: Float[Tensor, "n_atoms 3"]) -> Float[Tensor, ""]:
        """Soft clashscore of one structure: logistic overlap count per 1000 heavy atoms.

        Parameters
        ----------
        coordinates
            Atomic coordinates in model atom order.

        Returns
        -------
        Tensor
            Scalar soft clashscore.
        """
        topology = self._clash
        positions = coordinates[topology.atoms]
        # Pairs past this overlap contribute < sigmoid(-5) and are not worth tracking.
        reach = 2 * topology.radii.max() - CLASH_OVERLAP + 5 * CLASH_WIDTH
        i, j = close_pairs(positions, reach, self.clash_chunk_size)
        near_in_sequence = (topology.chain[i] == topology.chain[j]) & (
            (topology.residue[i] - topology.residue[j]).abs() <= 1
        )
        disulfide = topology.disulfide_sulfur[i] & topology.disulfide_sulfur[j]
        keep = ~near_in_sequence & ~disulfide
        if self.skip_polar_pairs:
            keep &= ~(topology.polar[i] & topology.polar[j])
        i, j = i[keep], j[keep]
        overlap = topology.radii[i] + topology.radii[j] - (positions[i] - positions[j]).norm(dim=-1)
        clashes = torch.sigmoid((overlap - CLASH_OVERLAP) / CLASH_WIDTH).sum()
        return 1000.0 * clashes / len(topology.atoms)

    def ramachandran_outside_favored(
        self, coordinates: Float[Tensor, "batch n_atoms 3"]
    ) -> Float[Tensor, " batch"]:
        """Soft percentage of residues outside the Ramachandran favored regions.

        Parameters
        ----------
        coordinates
            Atomic coordinates in model atom order.

        Returns
        -------
        Tensor
            Percentage per batch member.
        """
        if len(self._rama_table) == 0:
            return coordinates.new_zeros(coordinates.shape[0])
        backbone = coordinates[:, self._rama_atoms]  # [batch, n_residues, 5, 3]
        phi_psi = torch.stack(
            [dihedral(backbone[..., 0:4, :]), dihedral(backbone[..., 1:5, :])], dim=-1
        )
        omega = dihedral(
            torch.stack(
                [
                    coordinates[:, self._rama_previous_ca],
                    backbone[..., 0, :],
                    backbone[..., 1, :],
                    backbone[..., 2, :],
                ],
                dim=-2,
            )
        )
        table = torch.where(
            (self._rama_table == _TRANS_PROLINE) & (omega.abs() < 30.0),
            _CIS_PROLINE,
            self._rama_table,
        )
        values = phi_psi.new_zeros(phi_psi.shape[:-1])
        for index, name in enumerate(_RAMACHANDRAN_TABLES):
            values = torch.where(table == index, self._grids[("rama", name)](phi_psi), values)
        outside = soft_below(values, RAMACHANDRAN_FAVORED, self.outlier_softness)
        return 100.0 * outside.mean(dim=-1)

    def rotamer_outliers(
        self, coordinates: Float[Tensor, "batch n_atoms 3"]
    ) -> Float[Tensor, " batch"]:
        """Soft percentage of side chains that are rotamer outliers.

        Parameters
        ----------
        coordinates
            Atomic coordinates in model atom order.

        Returns
        -------
        Tensor
            Percentage per batch member.
        """
        outliers, count = coordinates.new_zeros(coordinates.shape[0]), 0
        for table, quadruples in self._rotamer_atoms.items():
            chis = torch.remainder(dihedral(coordinates[:, quadruples]), 360.0)
            if table in _HALF_TURN_LAST_CHI:
                chis = torch.cat([chis[..., :-1], torch.remainder(chis[..., -1:], 180.0)], -1)
            values = self._grids[("rota", table)](chis)
            outliers = outliers + soft_below(values, ROTAMER_OUTLIER, self.outlier_softness).sum(-1)
            count += len(quadruples)
        return 100.0 * outliers / max(count, 1)

    def __call__(
        self,
        coordinates: Float[Tensor, "batch n_atoms 3"],
        elements: Int[Tensor, "batch n_atoms"],
        b_factors: Float[Tensor, "batch n_atoms"],
        occupancies: Float[Tensor, "batch n_atoms"],
    ) -> Float[Tensor, ""]:
        """Soft MolProbity score of ``coordinates``, summed over the batch.

        Parameters
        ----------
        coordinates
            Atomic coordinates in model atom order, shape [batch, n_atoms, 3].
        elements, b_factors, occupancies
            Accepted for ``LossFunctionProtocol`` compatibility and ignored.

        Returns
        -------
        Float[Tensor, ""]
            Summed score.

        Raises
        ------
        RuntimeError
            If called before :meth:`prepare`.
        """
        if not self._prepared:
            raise RuntimeError("Call prepare() before scoring with molprobity.")
        clash = torch.stack([self.clashscore(member) for member in coordinates])
        rotamer = self.rotamer_outliers(coordinates)
        ramachandran = self.ramachandran_outside_favored(coordinates)
        clash_weight, rotamer_weight, ramachandran_weight = _SCORE_WEIGHTS
        score = (
            clash_weight * torch.log1p(clash)
            + rotamer_weight * torch.log1p(torch.relu(rotamer - 1.0))
            + ramachandran_weight * torch.log1p(torch.relu(ramachandran - 2.0))
        )
        return score.sum()


def build_molprobity_loss(
    options: MolProbityOptions, context: LossBuildContext
) -> MolProbityLossFunction:
    """Build the MolProbity surrogate from its configured options.

    Parameters
    ----------
    options
        See :class:`~sampleworks.core.losses.options.MolProbityOptions`.
    context
        Run-level inputs. Unused: the topology and device arrive in ``prepare()``.

    Returns
    -------
    MolProbityLossFunction
        Unprepared penalty.
    """
    del context
    return MolProbityLossFunction(
        outlier_softness=options.outlier_softness,
        clash_chunk_size=options.clash_chunk_size,
        skip_polar_pairs=options.skip_polar_pairs,
    )
