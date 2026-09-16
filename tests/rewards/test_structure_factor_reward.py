"""Tests specific to the structure-factor (reciprocal-space) reward function.

Reward-agnostic contract tests (interface, relative correlation, coordinate gradients,
batch=1 occupancy gradients, batch/edge handling) live in
``test_reward_function_contract.py``, where they run against every reward. What remains
here is specific to ``StructureFactorRewardFunction``
(``sampleworks.core.rewards.structure_factor``) or to SFcalculator's forward model.

Specific features of the SF reward include:
- No per-conformer occupancy/B-factors — SFcalculator batches only coordinates, so
  ``__call__`` should reject if not broadcastable across the batch.
- Bulk-solvent modes — ``off`` scores ``|Fprotein|``; ``combined`` or ``per_conformer``
  additionally account for bulk solvent contributions depending on the ensemble.
- MTZ information — unit cell, space group, and amplitude/sigma column; reflection masking
  for outliers and R-free test set.
"""

import logging
from pathlib import Path
from unittest import mock

import gemmi
import numpy as np
import pytest
import reciprocalspaceship as rs
import torch
from biotite.structure import AtomArray
from reciprocalspaceship.utils import add_rfree
from sampleworks.core.rewards.protocol import RewardInputs
from sampleworks.core.rewards.structure_factor import (
    _MIN_RETAINED_REFLECTION_FRACTION,
    _MIN_RETAINED_REFLECTIONS,
    StructureFactorRewardFunction,
)
from sampleworks.utils.atom_array_utils import (
    build_pairwise_altloc_arrays,
    find_all_altloc_ids,
)

from tests.rewards.reward_input_helpers import build_reward_input_tensors_without_coords


def make_prepared_reward(mtz_path, atom_array, device: torch.device, **kwargs):
    """Construct and ``prepare()`` an SF reward with a per-test config.

    Touches the SFcalculator forward model (heavier than a header read), so callers are marked
    ``gpu``. ``kwargs`` pass straight to the constructor (e.g. ``bulk_solvent``,
    ``normalize_amplitude``, ``exclude_free_reflections``, ``expcolumns``). The single fixed
    config lives in the ``reward_function_1vme_sf`` fixture; these tests need varied configs.

    ``prepare()`` takes reward inputs, so the SFcalculator topology is ``atom_array`` wrapped as
    single-conformer ``RewardInputs`` (the coordinates and B-factors are the array's own).
    """
    rf = StructureFactorRewardFunction(mtz_path, **kwargs)
    rf.prepare(
        RewardInputs.from_atom_array(atom_array, ensemble_size=1, device=device), device=device
    )
    return rf


def write_toy_mtz(
    path: Path,
    *,
    cell: gemmi.UnitCell,
    spacegroup: str,
    column_pairs: tuple[tuple[str, str], ...],
) -> Path:
    """Write a minimal MTZ carrying only header metadata, and return ``path``.

    Five toy reflections written for the reward function's construction-time tests. Each entry
    of ``column_pairs`` becomes one ``(amplitude, sigma)`` pair, amplitudes scaled per pair so
    the sets are distinguishable.

    Parameters
    ----------
    path
        Destination ``.mtz`` file.
    cell
        Unit cell written to the CELL header. Pass a default-constructed ``gemmi.UnitCell()``
        to emulate an MTZ whose cell is gemmi's placeholder.
    spacegroup
        Space group as a Hermann-Mauguin string. Required: gemmi refuses to write an MTZ with
        no space group ("Cannot write Mtz which has no space group") and rs raises before that
        ("has no space group information"), so there is no ``None`` to pass here.
    column_pairs
        One ``(amplitude, sigma)`` name pair per structure-factor set to write.

    Returns
    -------
    Path
        The ``path`` that was written, for convenient use as a fixture return value.
    """
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [2, 1, 0]], dtype=np.int32)
    amplitudes = np.array([100.0, 90.0, 80.0, 70.0, 60.0], dtype=np.float32)
    columns = {"H": hkl[:, 0], "K": hkl[:, 1], "L": hkl[:, 2]}
    for pair_index, (amplitude_col, sigma_col) in enumerate(column_pairs):
        columns[amplitude_col] = amplitudes * (1.0 + 0.1 * pair_index)
        columns[sigma_col] = amplitudes / 10.0
    dataset = rs.DataSet(columns, cell=cell, spacegroup=spacegroup).infer_mtz_dtypes()
    dataset.set_index(["H", "K", "L"], inplace=True)
    dataset.write_mtz(str(path))
    return path


@pytest.fixture(scope="module")
def sf_true_inputs(test_coordinates_1vme_sf, device: torch.device) -> dict[str, torch.Tensor]:
    """``__call__`` kwargs (batch=1) for the true 1vme structure.

    The reward-agnostic contract tests already exercise the generic interface/gradient/batch
    behavior against the SF reward's default config, so the tests here only add the
    config-specific deltas and reuse these inputs.
    """
    coords, atom_array = test_coordinates_1vme_sf
    elements, b_factors, occupancies = build_reward_input_tensors_without_coords(atom_array, device)
    return dict(
        coordinates=coords.unsqueeze(0),
        elements=elements.unsqueeze(0),
        b_factors=b_factors.unsqueeze(0),
        occupancies=occupancies.unsqueeze(0),
    )


@pytest.fixture(scope="module")
def sf_ensemble_inputs(
    structure_1vme_sf, device: torch.device
) -> tuple[dict[str, torch.Tensor], AtomArray]:
    """A 2-conformer ensemble (altloc A vs B) for the 1vme structure.

    The batch=2 counterpart to `sf_true_inputs`, returned as ``(__call__ kwargs, atom array)``
    tuple. The atom array is used for ``prepare()`` to build the SFcalculator.

    ``build_pairwise_altloc_arrays`` pairs each altloc with the shared blank-altloc atoms and
    filters to their common atoms, so both frames share one topology and differ *only* in the
    alternate-conformation coordinates. Residues modeled in only one altloc have no counterpart
    and are dropped to keep that shared topology. This divergence is what separates behavior
    that is nonlinear in the per-conformer density from behavior that is not — for example, the
    different bulk solvent modes.

    SFcalculator has no per-conformer occ/B axis, so ``b_factors`` is shared from altloc-A
    across the batch, and occupancy shared at the uniform ``1/batch_size = 0.5``.
    """
    altloc_ids = sorted(find_all_altloc_ids(structure_1vme_sf))  # ["A", "B"]
    altloc_a, altloc_b = altloc_ids
    altloc_pairs = build_pairwise_altloc_arrays(structure_1vme_sf, altloc_ids)
    array_a, array_b = altloc_pairs[(altloc_a, altloc_b)]
    ref_atom_array = array_a[0]  # state-A reference; retains element/b_factor/occupancy

    coords = torch.stack(
        [
            torch.from_numpy(array_a.coord[0]),  # altloc-A conformer [N, 3]
            torch.from_numpy(array_b.coord[0]),  # altloc-B conformer [N, 3]
        ]
    ).to(device=device, dtype=torch.float32)  # [2, N, 3]

    elements, b_factors, _ = build_reward_input_tensors_without_coords(ref_atom_array, device)
    n_atoms = coords.shape[1]
    occ = torch.full((n_atoms,), 0.5, device=device)  # uniform 1/batch_size
    reward_inputs = dict(
        coordinates=coords,
        elements=elements.unsqueeze(0).expand(2, -1),
        b_factors=b_factors.unsqueeze(0).expand(2, -1),
        occupancies=occ.unsqueeze(0).expand(2, -1),
    )
    return reward_inputs, ref_atom_array


class TestStructureFactorConstruction:
    """Construction-time behavior, on CPU: no GPU and no ``prepare()``/SF compute.

    ``__init__`` validates its config and reads exactly three things from the MTZ
    (``_resolve_mtz_metadata``): the unit cell, the space group, and the amplitude/sigma column
    layout. For efficiency, we build a ``toy_multi_set_mtz`` instead of taking the session-scoped
    ``mtz_path_1vme`` that on a CPU costs ~18 s, and build it once per class since several tests
    here only read it. The MTZ metadata are specified as class constants because they should be
    immutable for assertions.
    """

    # a, b, c (A), alpha, beta, gamma (deg) — a gemmi.UnitCell is built from these on demand.
    TOY_CELL_PARAMETERS = (11.0, 22.0, 33.0, 90.0, 100.0, 120.0)
    TOY_SPACE_GROUP = "P 1 2 1"  # Hermann-Mauguin string
    PROTEIN_COLUMNS = ("Fprotein", "SIGFprotein")  # (amplitude, sigma), protein-only
    TOTAL_COLUMNS = ("Ftotal", "SIGFtotal")  # (amplitude, sigma), protein + bulk solvent

    @pytest.fixture(scope="class")
    @classmethod
    def toy_multi_set_mtz(cls, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """A minimal multi-set MTZ carrying header metadata on unit cell and space group."""
        return write_toy_mtz(
            tmp_path_factory.mktemp("sf_construction") / "toy_multi_set.mtz",
            cell=gemmi.UnitCell(*cls.TOY_CELL_PARAMETERS),
            spacegroup=cls.TOY_SPACE_GROUP,
            column_pairs=(cls.PROTEIN_COLUMNS, cls.TOTAL_COLUMNS),
        )

    # The sibling test for mtz without space-group (_resolve_mtz_metadata raises when MTZ
    # has space group None) is not tested, because gemmi / rs prevents writing such an MTZ,
    # although such a MTZ could exist from other programs or corrupted files.
    def test_mtz_without_unit_cell_raises(self, tmp_path: Path):
        """An MTZ left with gemmi's placeholder cell is rejected at construction.

        Unlike the space group, this input *is* constructible: a default ``gemmi.UnitCell`` is
        (1, 1, 1, 90, 90, 90), which both writers accept and ``is_crystal()`` reports as False.
        """
        mtz_path = write_toy_mtz(
            tmp_path / "no_unit_cell.mtz",
            cell=gemmi.UnitCell(),
            spacegroup=self.TOY_SPACE_GROUP,
            column_pairs=(self.PROTEIN_COLUMNS,),
        )
        with pytest.raises(ValueError, match="carries no unit cell"):
            StructureFactorRewardFunction(mtz_path)

    def test_multi_set_mtz_requires_expcolumns(self, toy_multi_set_mtz):
        """A multi-set MTZ with no ``expcolumns`` is ambiguous and fails fast, rather than
        silently auto-selecting the first amplitude/sigma pair.
        """
        with pytest.raises(ValueError, match="Multiple SFAmplitude columns"):
            StructureFactorRewardFunction(toy_multi_set_mtz)

    def test_explicit_expcolumns_override_selection(self, toy_multi_set_mtz):
        """Explicit ``expcolumns`` are used verbatim, overriding the auto-selected first pair."""
        reward = StructureFactorRewardFunction(
            toy_multi_set_mtz, expcolumns=list(self.TOTAL_COLUMNS)
        )
        assert reward.expcolumns == list(self.TOTAL_COLUMNS)

    def test_unknown_expcolumns_raise(self, toy_multi_set_mtz):
        """Explicit ``expcolumns`` naming a column absent from the MTZ fail fast at construction."""
        with pytest.raises(ValueError, match="is not among the dataset's SFAmplitude columns"):
            StructureFactorRewardFunction(
                toy_multi_set_mtz, expcolumns=["Fnonexistent", self.PROTEIN_COLUMNS[1]]
            )

    @pytest.mark.parametrize(
        "bad_expcolumns",
        [
            ["Fprotein"],  # missing the sigma
            ["Fprotein", "SIGFprotein", "Ftotal"],  # too many
            [],
            "FP",  # a bare string of length 2 would otherwise split into ["F", "P"]
        ],
    )
    def test_malformed_expcolumns_raise(self, toy_multi_set_mtz, bad_expcolumns):
        """``expcolumns`` that is not an ``[amplitude, sigma]`` pair fails fast, including a
        bare string, which would otherwise be indexed character-wise into a bogus column pair.
        """
        with pytest.raises(ValueError, match=r"expcolumns must be a \[amplitude, sigma\] pair"):
            StructureFactorRewardFunction(toy_multi_set_mtz, expcolumns=bad_expcolumns)

    def test_cell_and_space_group_read_from_mtz(self, toy_multi_set_mtz):
        """The cell and space group are parsed off the MTZ, the only source for them."""
        reward = StructureFactorRewardFunction(
            toy_multi_set_mtz, expcolumns=list(self.TOTAL_COLUMNS)
        )
        assert reward.space_group == self.TOY_SPACE_GROUP
        assert reward.unit_cell.parameters == pytest.approx(self.TOY_CELL_PARAMETERS, abs=1e-3)

    @pytest.mark.parametrize("bad_partition", [0, -5, 10.0, 2.5, "10", None, True])
    def test_invalid_batch_partition_raises(self, toy_multi_set_mtz, bad_partition):
        """A non-integer or non-positive ``batch_partition`` (an OOM knob) fails fast.

        The check precedes column resolution, so no ``expcolumns`` are needed.
        """
        with pytest.raises(ValueError, match="batch_partition must be a positive integer"):
            StructureFactorRewardFunction(toy_multi_set_mtz, batch_partition=bad_partition)

    def test_unknown_bulk_solvent_mode_raises(self, toy_multi_set_mtz):
        """An unrecognized ``bulk_solvent`` mode fails fast at construction."""
        with pytest.raises(ValueError, match="bulk_solvent must be one of"):
            StructureFactorRewardFunction(toy_multi_set_mtz, bulk_solvent="not_off")

    def test_reserved_sfcalculator_kwargs_raise(self, toy_multi_set_mtz):
        """``sfcalculator_kwargs`` should not override reserved keys for the reward function's
        ``__init__ arguments``."""
        with pytest.raises(ValueError, match="may not override reserved keys"):
            StructureFactorRewardFunction(
                toy_multi_set_mtz,
                expcolumns=list(self.TOTAL_COLUMNS),
                sfcalculator_kwargs={"dmin": 1.0, "device": "cuda"},
            )

    def test_call_before_prepare_raises(self, toy_multi_set_mtz):
        """Evaluating the reward before ``prepare()`` gives a clear error."""
        reward = StructureFactorRewardFunction(
            toy_multi_set_mtz, expcolumns=list(self.TOTAL_COLUMNS)
        )
        n_atoms = 4
        with pytest.raises(RuntimeError, match=r"prepare\(\) must be called"):
            reward(
                coordinates=torch.zeros(1, n_atoms, 3),
                elements=torch.zeros(1, n_atoms, dtype=torch.int64),
                b_factors=torch.full((1, n_atoms), 20.0),
                occupancies=torch.ones(1, n_atoms),
            )


@pytest.mark.gpu
class TestStructureFactorOccupancy:
    """``__call__``'s input guards, both of which run before any SF compute: the atom count must
    match the topology fixed by ``prepare()``, and occupancy/B must be broadcast-identical across
    the batch (SFC has no per-conformer occupancy/B axis).
    """

    def test_atom_count_mismatch_raises(self, reward_function_1vme_sf, sf_ensemble_inputs):
        """Inputs whose atom count disagrees with the prepared topology fail fast.

        ``reward_function_1vme_sf`` is prepared on the full 1vme topology (both altlocs present),
        while ``sf_ensemble_inputs`` is built on the shared-altloc topology (blank plus the atoms
        common to A and B), which is strictly smaller.
        """
        ensemble_reward_inputs, _ = sf_ensemble_inputs
        with pytest.raises(ValueError, match="topology built by"):
            reward_function_1vme_sf(**ensemble_reward_inputs)

    @pytest.mark.parametrize("field", ["occupancies", "b_factors"])
    def test_per_conformer_occupancy_or_b_raises(
        self, reward_function_1vme_sf, test_coordinates_1vme_sf, device, field
    ):
        """Per-conformer (non-broadcast) occupancy/B is rejected, not silently dropped.

        SFcalculator batches only coordinates, so __call__ honors a single shared occupancy/B
        vector. Production always feeds broadcast-identical rows (the batch=1 and identical-row
        ensemble cases are covered by the shared contract tests); this pins that genuinely
        per-conformer values raise instead of being silently ignored. The guard runs before any
        SF compute, so this also documents that the [batch, n_atoms] signature does NOT mean
        per-conformer occupancy/B is supported.
        """
        coords, atom_array = test_coordinates_1vme_sf
        elements, b_factors, occupancies = build_reward_input_tensors_without_coords(
            atom_array, device
        )

        batch = 3
        kwargs = dict(
            coordinates=coords.unsqueeze(0).expand(batch, -1, -1),
            elements=elements.unsqueeze(0).expand(batch, -1),
            b_factors=b_factors.unsqueeze(0).expand(batch, -1),
            occupancies=occupancies.unsqueeze(0).expand(batch, -1),
        )
        # Make one conformer's values genuinely differ (additive shift works regardless of zeros).
        perturbed = kwargs[field].clone()
        perturbed[1] += 1.0
        kwargs[field] = perturbed

        with pytest.raises(ValueError, match="identical across the batch"):
            reward_function_1vme_sf(**kwargs)


@pytest.mark.gpu
class TestStructureFactorBulkSolvent:
    """The bulk-solvent modes. ``off`` scores ``|Fprotein|`` (covered by the contract tests);
    ``combined``/``per_conformer`` score ``|Ftotal|`` with default-scaled solvent. The two
    ``|Ftotal|`` modes differ only for a real ensemble. Comparisons are relative, so they don't
    depend on the raw ``|F|`` scale.
    """

    # Four prepared Ftotal rewards: {combined, per_conformer} x {full, ensemble} topology, all
    # raw |F| and all scored against the MTZ's Ftotal set. The topology differs on atom counts:
    #   _full     -- a single 1vme structure with both altlocs present
    #   _ensemble -- topology consists of blank + shared atoms between the two altlocs
    @pytest.fixture(scope="class")
    @classmethod
    def reward_with_solvent_combined_full(cls, mtz_path_1vme, structure_1vme_sf, device):
        """``combined`` on the full 1vme topology."""
        return make_prepared_reward(
            mtz_path_1vme,
            structure_1vme_sf,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="combined",
            normalize_amplitude=False,
        )

    @pytest.fixture(scope="class")
    @classmethod
    def reward_with_solvent_per_conformer_full(cls, mtz_path_1vme, structure_1vme_sf, device):
        """``per_conformer`` on the full 1vme topology."""
        return make_prepared_reward(
            mtz_path_1vme,
            structure_1vme_sf,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="per_conformer",
            normalize_amplitude=False,
        )

    @pytest.fixture(scope="class")
    @classmethod
    def reward_with_solvent_combined_ensemble(cls, mtz_path_1vme, sf_ensemble_inputs, device):
        """``combined`` on the altloc-ensemble topology."""
        _, ref_atom_array = sf_ensemble_inputs
        return make_prepared_reward(
            mtz_path_1vme,
            ref_atom_array,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="combined",
            normalize_amplitude=False,
        )

    @pytest.fixture(scope="class")
    @classmethod
    def reward_with_solvent_per_conformer_ensemble(cls, mtz_path_1vme, sf_ensemble_inputs, device):
        """``per_conformer`` on the altloc-ensemble topology."""
        _, ref_atom_array = sf_ensemble_inputs
        return make_prepared_reward(
            mtz_path_1vme,
            ref_atom_array,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="per_conformer",
            normalize_amplitude=False,
        )

    def test_ftotal_modes_agree_for_single_conformer(
        self,
        reward_with_solvent_combined_full,
        reward_with_solvent_per_conformer_full,
        sf_true_inputs,
    ):
        """For a single conformer (batch_size=1) ``mask(<rho>)`` and ``<mask(rho)>`` are the same
        mask, so ``combined`` and ``per_conformer`` give the same loss."""
        torch.testing.assert_close(
            reward_with_solvent_combined_full(**sf_true_inputs),
            reward_with_solvent_per_conformer_full(**sf_true_inputs),
        )

    def test_combined_fits_ftotal_column(
        self,
        mtz_path_1vme,
        structure_1vme_sf,
        reward_with_solvent_combined_full,
        sf_true_inputs,
        device,
    ):
        """Adding default-scaled bulk solvent (``combined``) fits the synthetic ``Ftotal``
        column far better than protein-only (``off``). It should also reproduce ``Ftotal``
        as the synthetic MTZ should be generated with the same structure and config."""
        reward_with_solvent_off = make_prepared_reward(
            mtz_path_1vme,
            structure_1vme_sf,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="off",
            normalize_amplitude=False,
        )
        loss_combined = reward_with_solvent_combined_full(**sf_true_inputs)
        loss_off = reward_with_solvent_off(**sf_true_inputs)
        assert loss_combined < loss_off
        assert loss_combined.item() < 1e-6, (
            "combined mode should reproduce MTZ's Ftotal with loss less than 1e-6, but the "
            f"combined loss here is {loss_combined.item():.6e}"
        )

    def test_per_conformer_averages_masks_over_ensemble(
        self, reward_with_solvent_per_conformer_full, sf_true_inputs
    ):
        """batch_size=2 *identical* conformers (occ 1/2 each) score the same as the single
        conformer.

        This guards the equal-population assumption in ``per_conformer``: the protein path bakes
        ``atom_occ`` into each conformer and *sums*, while the solvent path averages
        scale-invariant per-conformer masks (``Fmask_HKL_batch.mean(dim=0)``) — a hardcoded
        uniform ``1/batch_size`` weight. Both are population *averages*, so occ = 1/batch_size
        keeps them consistent and the ensemble matches the single conformer; this would break if
        ``per_conformer`` summed (rather than averaged) the masks. Non-uniform per-conformer
        occupancy is properly rejected by the reward now — see
        ``TestStructureFactorOccupancy.test_per_conformer_occupancy_or_b_raises``.
        """
        identical_ensemble = dict(
            coordinates=sf_true_inputs["coordinates"].expand(2, -1, -1),
            elements=sf_true_inputs["elements"].expand(2, -1),
            b_factors=sf_true_inputs["b_factors"].expand(2, -1),
            occupancies=(sf_true_inputs["occupancies"] / 2).expand(2, -1),
        )
        torch.testing.assert_close(
            reward_with_solvent_per_conformer_full(**sf_true_inputs),
            reward_with_solvent_per_conformer_full(**identical_ensemble),
        )

    def test_ftotal_modes_diverge_for_distinct_conformer_ensemble(
        self,
        reward_with_solvent_combined_ensemble,
        reward_with_solvent_per_conformer_ensemble,
        sf_ensemble_inputs,
    ):
        """For a genuine 2-conformer ensemble (altloc A vs B), ``mask(<rho>) != <mask(rho)>``, so
        ``combined`` and ``per_conformer`` give *different* losses.

        This is the behavior that justifies keeping both modes — the agree-cases above
        (batch_size=1 and batch_size=2 *identical* conformers) never exercise the nonlinearity.
        Here the two frames differ in the alternate-conformation coordinates, so the combined
        mask (built from the summed density) and the per-conformer mean of masks genuinely
        diverge.

        The mock.patch.object supplements the numerical difference test by checking the dispatch
        logic and ensure that the correct SFcalculator method for Fsolvent is called.
        """
        ensemble_reward_inputs, _ = sf_ensemble_inputs
        loss_combined = reward_with_solvent_combined_ensemble(**ensemble_reward_inputs)
        # Specifically checking the dispatch logic in _compute_ensemble_ftotal, where
        # ``bulk_solvent="per_conformer"`` should route to a calc_fsolvent_batch call.
        with mock.patch.object(
            reward_with_solvent_per_conformer_ensemble.sfc,
            "calc_fsolvent_batch",
            wraps=reward_with_solvent_per_conformer_ensemble.sfc.calc_fsolvent_batch,
        ) as sfc_calc_fsolvent_batch:
            loss_per_conformer = reward_with_solvent_per_conformer_ensemble(
                **ensemble_reward_inputs
            )
        assert sfc_calc_fsolvent_batch.call_count == 1
        assert torch.isfinite(loss_combined) and torch.isfinite(loss_per_conformer)

        relative_gap = ((loss_per_conformer - loss_combined) / loss_combined).item()
        relative_threshold = 0.2
        # The true relative difference is ~1.24 and the noise from permuting the batch for
        # per_conformer is ~0.02. Subjected to change if loss function or normalization changes.
        assert abs(relative_gap) > relative_threshold, (
            f"combined and per_conformer losses differ relatively by {relative_gap:+.3f}, "
            f"under the {relative_threshold} threshold we set: {loss_combined.item()} vs "
            f"{loss_per_conformer.item()}"
        )

    @pytest.mark.parametrize(
        "reward_fixture",
        ["reward_with_solvent_combined_ensemble", "reward_with_solvent_per_conformer_ensemble"],
    )
    def test_bulk_solvent_branch_carries_gradient(
        self, request: pytest.FixtureRequest, reward_fixture: str, sf_ensemble_inputs
    ):
        """Fsolvent under ``combined`` and ``per_conformer`` modes should contribute gradients
        to the coordinates.

        ``Ftotal = kiso * aniso * (Fprotein_HKL + kmask * Fmask_HKL)``, and ``Fmask_HKL``
        descends from ``Fprotein_asu_batch`` rather than from ``Fprotein_HKL``.
        """
        reward = request.getfixturevalue(reward_fixture)
        ensemble_reward_inputs, _ = sf_ensemble_inputs
        coordinates = ensemble_reward_inputs["coordinates"].clone().requires_grad_(True)
        loss = reward(**{**ensemble_reward_inputs, "coordinates": coordinates})

        Fprotein_HKL, Fmask_HKL = reward.sfc.Fprotein_HKL, reward.sfc.Fmask_HKL
        assert Fprotein_HKL.requires_grad and Fmask_HKL.requires_grad, (
            f"{reward.bulk_solvent} mode should have requires_grad=True on both Fprotein_HKL "
            f"and Fmask_HKL, but got {Fprotein_HKL.requires_grad} and {Fmask_HKL.requires_grad}"
        )
        (total_gradient,) = torch.autograd.grad(loss, coordinates, retain_graph=True)
        d_protein, d_solvent = torch.autograd.grad(
            loss, (Fprotein_HKL, Fmask_HKL), retain_graph=True
        )
        (via_protein,) = torch.autograd.grad(
            Fprotein_HKL, coordinates, grad_outputs=d_protein, retain_graph=True
        )
        (via_solvent,) = torch.autograd.grad(Fmask_HKL, coordinates, grad_outputs=d_solvent)

        for branch, gradient in (("Fprotein", via_protein), ("Fmask", via_solvent)):
            assert 0 < gradient.norm().item() < 1e6, (
                f"the coordinate gradient through {branch} in {reward.bulk_solvent} mode has "
                f"norm {gradient.norm().item():.3e}"
            )

        residual = (
            (via_protein + via_solvent - total_gradient).norm() / total_gradient.norm()
        ).item()
        assert residual < 1e-3, (
            "coordinate gradients from the Fprotein and Fmask branches do not sum back to "
            f"d(loss)/d(coords) (relative residual norm {residual:.2e})"
        )


@pytest.mark.gpu
class TestStructureFactorConfig:
    """Config knobs beyond the forward model: reflection selection."""

    def test_exclude_free_set_drops_reflections(self, mtz_path_1vme, structure_1vme_sf, device):
        """``exclude_free_reflections=True`` drops the R-free test set from the scored mask.

        The synthetic MTZ's free column is ``R-free-flags`` with the test set flagged 1
        (rs/Phenix convention), so SFcalculator is pointed at it explicitly — its defaults
        (``FreeR_flag`` / testset value 0) don't match. Outliers are always excluded regardless.
        """
        free_flag_kwargs = {"freeflag": "R-free-flags", "testset_value": 1}
        reward_all = make_prepared_reward(
            mtz_path_1vme,
            structure_1vme_sf,
            device,
            expcolumns=["Fprotein", "SIGFprotein"],
            exclude_free_reflections=False,
            sfcalculator_kwargs=free_flag_kwargs,
        )
        reward_work = make_prepared_reward(
            mtz_path_1vme,
            structure_1vme_sf,
            device,
            expcolumns=["Fprotein", "SIGFprotein"],
            exclude_free_reflections=True,
            sfcalculator_kwargs=free_flag_kwargs,
        )
        assert reward_all.sfc.free_flag.any()  # safety: the free set is actually recognized
        assert int(reward_work._reflection_mask.sum()) < int(reward_all._reflection_mask.sum())

    def test_empty_reflection_mask_raises(
        self, mtz_path_1vme, structure_1vme_sf, device, tmp_path: Path
    ):
        """A mask that scores no reflection fails in ``prepare()``.

        The all-free MTZ is built with rs's own ``add_rfree`` (``fraction=1.0``) so the flag
        column name and convention match the committed MTZ's (``R-free-flags``, test set = 1).
        """
        dataset = add_rfree(rs.read_mtz(str(mtz_path_1vme)), fraction=1.0, seed=0)
        mtz_all_free = tmp_path / "1vme_all_free.mtz"
        dataset.write_mtz(str(mtz_all_free))
        with pytest.raises(ValueError, match="No reflections remain"):
            make_prepared_reward(
                mtz_all_free,
                structure_1vme_sf,
                device,
                expcolumns=["Fprotein", "SIGFprotein"],
                exclude_free_reflections=True,
                sfcalculator_kwargs={"freeflag": "R-free-flags", "testset_value": 1},
            )

    def test_inverted_testset_value_warns(self, mtz_path_1vme, structure_1vme_sf, device, caplog):
        """An inverted ``testset_value`` keeps only the test set, and is warned about.

        Inverting the flag keeps the fraction of valid reflections small although the
        absolute size can still be large.
        """
        with caplog.at_level(logging.WARNING):
            reward = make_prepared_reward(
                mtz_path_1vme,
                structure_1vme_sf,
                device,
                expcolumns=["Fprotein", "SIGFprotein"],
                exclude_free_reflections=True,
                sfcalculator_kwargs={"freeflag": "R-free-flags", "testset_value": 0},
            )
        n_used, n_total = int(reward._reflection_mask.sum()), len(reward.sfc.Fo)
        # check that the fraction precondition held and the floor's precondition did not hold
        assert n_used < _MIN_RETAINED_REFLECTION_FRACTION * n_total
        assert n_used >= _MIN_RETAINED_REFLECTIONS
        assert f"Only {n_used}/{n_total} reflections remain" in caplog.text
        # check that only the warning message from the fraction threshold is logged
        assert "testset_value" in caplog.text
        assert "the MTZ reflection range" not in caplog.text

    def test_small_reflection_set_warns(self, mtz_path_1vme, structure_1vme_sf, device, caplog):
        """A reflection set too small of absolute size to guide coordinates should warn.

        Truncating to 10 A leaves only a few hundred reflections but the fraction of valid
        reflections can still be large.
        """
        with caplog.at_level(logging.WARNING):
            reward = make_prepared_reward(
                mtz_path_1vme,
                structure_1vme_sf,
                device,
                expcolumns=["Fprotein", "SIGFprotein"],
                resolution=10.0,
            )
        n_used, n_total = int(reward._reflection_mask.sum()), len(reward.sfc.Fo)
        # check that the floor precondition held and the fraction's precondition did not hold
        assert n_used < _MIN_RETAINED_REFLECTIONS
        assert n_used >= _MIN_RETAINED_REFLECTION_FRACTION * n_total
        # check that only the warning message from the absolute threshold is logged
        assert "the MTZ reflection range" in caplog.text
        assert f"{n_used}/{n_total}" not in caplog.text
