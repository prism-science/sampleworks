"""Coordinate-space bond-geometry regularizer: bond-length and steric-clash hinges.

When a data loss is optimized aggressively, guidance can distort the denoised structure --
stretching covalent bonds and driving non-bonded atoms into steric clashes -- to buy a little data
fit. That is the overshoot observed on high-baseline targets (e.g. 1VME, where IT-opt
z-optimization raised clashes while dropping RSCC). This module adds a differentiable penalty on the
denoised coordinates that guidance must trade against, so it cannot reach a good data score
through broken geometry.

It is a faithful port of the reference ``BondLengthLossFunction``
(https://github.com/sai-advaith/it_opt, ``protenix/src/losses/bond_length_loss_function.py``):
a bonded-pair length hinge plus a non-bonded steric-clash hinge. Both take the violation's positive
part; the bond term then raises it to ``bond_power`` (default 2) while the clash term stays linear,
matching the reference. The one deliberate divergence is how ``collision_loss`` reduces the
ensemble axis; see the comment there. It is registered as the ``bond_geometry`` loss, so it combines
with a data loss through a loss configuration file, and ``LatentOptimization`` adds it to its
per-step loss directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from biotite.structure import AtomArray, connect_via_residue_names
from jaxtyping import Float, Int
from torch import Tensor

from sampleworks.utils.elements import covalent_radius


if TYPE_CHECKING:
    from sampleworks.core.losses.options import BondGeometryOptions
    from sampleworks.core.losses.protocol import LossInputs
    from sampleworks.core.losses.registry import LossBuildContext


# Type hints below (": str", "-> Tensor", "| None" meaning "or None") only CLAIM what a value
# should be -- nothing enforces them at runtime, and deleting them changes nothing. They are for
# human readers and the ty checker. Worked examples: core/scalers/latent_optimization.py.


class BondGeometryLoss:
    """Penalize distorted bonds and steric clashes in the denoised structure.

    A two-phase loss (see ``PreparableLossFunctionProtocol``): ``__init__`` takes the hinge
    settings, and :meth:`prepare` reads the topology (which atoms are bonded, their ideal bond
    lengths, the per-atom-pair collision distances, and which pairs to score for clashes) once from
    the model atom array, so each step is just cheap tensor ops on the coordinates. Only the
    coordinates are scored; the per-atom elements, B-factors and occupancies are ignored.

    The penalty is unweighted: a ``CompositeLoss`` weight, or the caller, scales it.
    """

    def __init__(
        self,
        *,
        bond_tolerance: float = 0.2,
        clash_padding: float = 0.4,
        bond_power: int = 2,
    ):
        """Store the hinge settings; the topology is built by :meth:`prepare`.

        ``bond_tolerance`` (Å) is the slack a bond may deviate from its ideal length before it is
        penalized; ``clash_padding`` (Å) is added to the sum of covalent radii to set how close
        non-bonded atoms may approach before they count as clashing; ``bond_power`` is the exponent
        on the bond-length hinge (2 = quadratic, per the reference).
        """
        self.bond_tolerance = bond_tolerance
        self.clash_padding = clash_padding
        self.bond_power = bond_power
        self._radii: Tensor | None = None

    def prepare(self, loss_inputs: LossInputs, *, device: torch.device | str = "cpu") -> None:
        """Build the bond topology and collision-distance matrix from the model atom array.

        Side effect: replaces this loss's topology tensors, so re-running it rebinds the loss.

        Parameters
        ----------
        loss_inputs
            Inputs for the atoms the model operates on; ``loss_inputs.atom_array`` sets the atom
            order of the coordinates ``__call__`` scores.
        device
            Torch device the topology tensors are placed on.

        Raises
        ------
        ValueError
            If ``loss_inputs`` carries no atom array.
        """
        atom_array = loss_inputs.atom_array
        if atom_array is None:
            raise ValueError(
                "BondGeometryLoss needs the model atom array to infer bonds; build the loss "
                "inputs with LossInputs.from_atom_array."
            )
        # One covalent radius per atom, looked up once. Both penalties below are sums of two radii,
        # so they index this instead of querying gemmi again per bond and per atom.
        self._radii = self._build_radii(atom_array, device)

        # Each bond connects atom a to atom b and has an ideal length; these three arrays line up,
        # one entry per bond.
        self._bond_atom_a, self._bond_atom_b, self._bond_lengths = self._build_bonds(
            atom_array, device
        )
        self._collision_distances = self._build_collision_distances()
        self._scored_pairs = self._build_scored_pairs(len(atom_array), device)

    # ================================ topology (built once) ================================

    def _build_radii(self, atom_array: AtomArray, device) -> Tensor:
        """Covalent radius of every atom, in atom-array order: [n_atoms].

        A bond's ideal length is the sum of its two endpoints' radii, and a pair's collision
        distance is the sum of that pair's radii, so every number both penalties need is a sum of
        two entries of this tensor. Looking each atom up once here keeps the lookups to one pass.
        """
        radii = [covalent_radius(element) for element in atom_array.element]
        return torch.tensor(radii, dtype=torch.float32, device=device)

    def _build_bonds(self, atom_array: AtomArray, device):
        """Bonded-atom endpoints and ideal lengths, inferred from residue-name templates.

        biotite's ``connect_via_residue_names`` infers the intra- and inter-residue bonds from the
        standard component templates and returns a ``BondList``. By default that list already holds
        each bond exactly once with the lower atom index first, so ``as_array()`` gives us the edge
        list directly -- its columns are atom i, atom j, bond type. (``get_all_bonds()`` would
        instead return padded per-atom adjacency, which lists every bond twice.) A bond's ideal
        length is the sum of the two atoms' covalent radii.

        Returns three aligned 1-D tensors, each of length n_bonds: the first endpoint of every bond,
        the second endpoint, and the ideal length.
        """
        bonds = connect_via_residue_names(atom_array).as_array()  # [n_bonds, 3]: i, j, bond type
        # as_array() is uint32, which torch refuses to convert, so cast to a signed integer.
        pair_array = bonds[:, :2].astype(np.int64)

        atom_a = torch.tensor(pair_array[:, 0], dtype=torch.long, device=device)
        atom_b = torch.tensor(pair_array[:, 1], dtype=torch.long, device=device)

        # Ideal length of a bond = sum of the two atoms' covalent radii.
        bond_lengths = self._radii[atom_a] + self._radii[atom_b]
        return atom_a, atom_b, bond_lengths

    def _build_collision_distances(self) -> Tensor:
        """The [n_atoms, n_atoms] matrix of minimum non-clashing distances (covalent-radii sums).

        Entry (i, j) is the sum of atom i's and atom j's covalent radii; two non-bonded atoms closer
        than this (plus ``clash_padding``) are overlapping.
        """
        # radius_i + radius_j for every pair: a column plus a row vector broadcasts to [n, n].
        radius_column = self._radii.unsqueeze(1)  # [n, 1]
        radius_row = self._radii.unsqueeze(0)  # [1, n]
        return radius_column + radius_row

    def _build_scored_pairs(self, n_atoms: int, device) -> Tensor:
        """Boolean [n_atoms, n_atoms] mask: which atom pairs count toward the clash penalty.

        We penalize every pair EXCEPT an atom with itself (distance 0) and directly-bonded atoms
        (they should sit close together). Built once from the fixed bond topology; the mask
        is symmetric, so each clashing pair is counted twice in the sum, matching the reference.
        """
        scored = torch.ones((n_atoms, n_atoms), dtype=torch.bool, device=device)
        # an atom never clashes with itself
        scored.fill_diagonal_(False)
        # directly-bonded atoms are supposed to sit close, so don't count them as clashes
        scored[self._bond_atom_a, self._bond_atom_b] = False
        scored[self._bond_atom_b, self._bond_atom_a] = False
        return scored

    # ============================== the two penalty terms ==============================

    def bond_length_loss(self, coords: Tensor) -> Tensor:
        """Hinge on bonded-pair length deviation beyond ``bond_tolerance``.

        ``coords`` is the denoised coordinate tensor, shape [ensemble_members, atoms, 3]. A bond is
        free within ``bond_tolerance`` of its ideal length; past that the deviation is
        raised to ``bond_power`` and summed over all bonds and ensemble members. Keeps the optimizer
        from stretching or compressing covalent bonds to chase density.
        """
        if self._bond_lengths.numel() == 0:
            return coords.new_zeros(())

        # Current length of every bond, for every ensemble member: [ensemble, n_bonds].
        pos_a = coords[:, self._bond_atom_a]
        pos_b = coords[:, self._bond_atom_b]
        lengths = (pos_a - pos_b).norm(dim=-1)

        # The deviation is unsigned, so a stretched and a compressed bond of equal error are
        # penalized alike. The clamp below is the tolerance window -- deviations under it are free
        # -- not a one-sided penalty like the one in collision_loss, which clamps a signed gap.
        deviation = (lengths - self._bond_lengths).abs()
        excess = (deviation - self.bond_tolerance).clamp(min=0)
        return excess.pow(self.bond_power).sum()

    def collision_loss(self, coords: Tensor) -> Tensor:
        """Hinge on non-bonded atoms closer than their collision distance + ``clash_padding``.

        ``coords`` is the denoised coordinate tensor, shape [ensemble, atoms, 3]. This is the
        steric-clash term: per atom pair we take the worst member's overlap plus the ensemble mean,
        then sum over the scored pairs. The reference used the worst member alone; the comment on
        the reduction below compares the four options and says why we add the mean. Note the
        pairwise-distance tensor is O(n_atoms**2), so memory scales with the square of the count.
        """
        # Distance between every pair of atoms, for each ensemble member: [ensemble, n, n].
        # torch.cdist gives the same distances without building the [ensemble, n, n, 3] grid of
        # coordinate differences a manual broadcast would (3x the distance matrix). Verified on the
        # pinned torch (2.7): its gradient is finite and matches the broadcast form even for
        # coincident atoms, so the historical cdist zero-distance NaN does not apply here.
        distances = torch.cdist(coords, coords)

        # How far each pair is inside its allowed distance (positive = overlapping). We clamp per
        # member, before reducing; the reference clamps after its max, which is equivalent there
        # (relu commutes with max) but not once the mean below is added, where a comfortably
        # separated member's negative gap would cancel another member's real overlap.
        min_distance = self._collision_distances + self.clash_padding
        overlap_per_member = (min_distance - distances).clamp(min=0)

        # Reduce the ensemble axis, then drop self- and bonded pairs. Four reductions were weighed;
        # the value column is relative to max, the reference's choice, with E ensemble members:
        #
        #   max          1x            only the worst member gets gradient -- the rest are ignored
        #   mean         1/E .. 1x     every member counted, but can weaken the penalty E-fold
        #   max + mean   1x .. 2x      every member counted, never weaker than max, capped at 2x
        #   sum          1x .. Ex      every member counted, but scales the penalty with E
        #
        # We take max + mean. The max keeps the worst member's full push, so the penalty can never
        # come out weaker than it is today -- which matters because the bond_geometry weight is
        # tuned to the smallest value that fixes clashes, and mean alone could drop below that
        # floor. The mean then gives every other clashing member a share instead of nothing, which
        # is what max alone failed to do. sum would do that too, but its value grows with ensemble
        # size.
        combined_overlap = overlap_per_member.max(dim=0).values + overlap_per_member.mean(dim=0)
        scored_overlap = combined_overlap * self._scored_pairs  # [n, n]; unscored pairs zeroed
        return scored_overlap.sum()

    def __call__(
        self,
        coordinates: Float[Tensor, "batch n_atoms 3"],
        elements: Int[Tensor, "batch n_atoms"],
        b_factors: Float[Tensor, "batch n_atoms"],
        occupancies: Float[Tensor, "batch n_atoms"],
    ) -> Float[Tensor, ""]:
        """Sum of the bond-length and collision penalties over the ensemble.

        Parameters
        ----------
        coordinates
            Denoised coordinates in model atom order, shape [batch, n_atoms, 3]; the batch axis is
            the ensemble.
        elements, b_factors, occupancies
            Accepted for ``LossFunctionProtocol`` compatibility and ignored.

        Returns
        -------
        Float[Tensor, ""]
            Scalar penalty to be minimized.

        Raises
        ------
        RuntimeError
            If :meth:`prepare` has not been called.
        """
        del elements, b_factors, occupancies
        if self._radii is None:
            raise RuntimeError("BondGeometryLoss.prepare() must be called before scoring.")
        return self.bond_length_loss(coordinates) + self.collision_loss(coordinates)


def build_bond_geometry_loss(
    options: BondGeometryOptions, context: LossBuildContext
) -> BondGeometryLoss:
    """Build the bond-geometry loss from its configured options.

    The topology arrives in :meth:`BondGeometryLoss.prepare`, once sampling knows the model atom
    array, so ``context`` is unused.

    Parameters
    ----------
    options
        Bond-geometry options; see :class:`~sampleworks.core.losses.options.BondGeometryOptions`.
    context
        Run-level inputs. Unused here, kept for a uniform builder signature.

    Returns
    -------
    BondGeometryLoss
        Unprepared loss penalizing bond-length deviations and steric clashes.
    """
    del context  # topology and device arrive in prepare()
    return BondGeometryLoss(
        bond_tolerance=options.bond_tolerance,
        clash_padding=options.clash_padding,
        bond_power=options.bond_power,
    )
