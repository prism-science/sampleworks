"""Integration tests for model/structure atom-count mismatch handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from biotite.structure import AtomArray
from sampleworks.core.rewards.protocol import RewardInputs
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.core.samplers.protocol import StepParams
from sampleworks.core.scalers.fk_steering import FKSteering
from sampleworks.core.scalers.pure_guidance import PureGuidance
from sampleworks.core.scalers.step_scalers import DataSpaceDPSScaler, NoiseSpaceDPSScaler
from sampleworks.eval.structure_utils import process_structure_to_trajectory_input
from sampleworks.utils.atom_array_utils import make_normalized_atom_id
from sampleworks.utils.atom_reconciler import AtomReconciler
from sampleworks.utils.frame_transforms import apply_forward_transform
from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import save_everything

from tests.mocks import MismatchCase, MismatchCaseWrapper
from tests.utils.atom_array_builders import build_test_atom_array


@dataclass(frozen=True)
class StructurePreprocessExpectation:
    """Expectations for real structure preprocessing.

    Parameters
    ----------
    id
        Case identifier used as pytest ID.
    fixture_name
        Fixture name providing the parsed structure dictionary.
    expected_filtered_atoms
        Expected atom count after occupancy/NaN filtering.
    expected_hydrogen_atoms
        Expected hydrogen count after filtering.
    """

    id: str
    fixture_name: str
    expected_filtered_atoms: int
    expected_hydrogen_atoms: int


STRUCTURE_PREPROCESS_EXPECTATIONS: tuple[StructurePreprocessExpectation, ...] = (
    StructurePreprocessExpectation(
        id="1vme_cif",
        fixture_name="structure_1vme",
        expected_filtered_atoms=6462,
        expected_hydrogen_atoms=0,
    ),
    StructurePreprocessExpectation(
        id="6b8x_pdb",
        fixture_name="structure_6b8x",
        expected_filtered_atoms=2435,
        expected_hydrogen_atoms=0,
    ),
    StructurePreprocessExpectation(
        id="2yl0_pdb",
        fixture_name="structure_2yl0",
        expected_filtered_atoms=1727,
        expected_hydrogen_atoms=727,
    ),
    StructurePreprocessExpectation(
        id="2yl0_cif",
        fixture_name="structure_2yl0_density",
        expected_filtered_atoms=955,
        expected_hydrogen_atoms=0,
    ),
    StructurePreprocessExpectation(
        id="5sop_pdb",
        fixture_name="structure_5sop",
        expected_filtered_atoms=4873,
        expected_hydrogen_atoms=2444,
    ),
    StructurePreprocessExpectation(
        id="5sop_cif",
        fixture_name="structure_5sop_density",
        expected_filtered_atoms=1264,
        expected_hydrogen_atoms=0,
    ),
    StructurePreprocessExpectation(
        id="6ni6_pdb",
        fixture_name="structure_6ni6",
        expected_filtered_atoms=6812,
        expected_hydrogen_atoms=3426,
    ),
    StructurePreprocessExpectation(
        id="6ni6_cif",
        fixture_name="structure_6ni6_density",
        expected_filtered_atoms=1678,
        expected_hydrogen_atoms=0,
    ),
    StructurePreprocessExpectation(
        id="9bn8_pdb",
        fixture_name="structure_9bn8",
        expected_filtered_atoms=3330,
        expected_hydrogen_atoms=0,
    ),
    StructurePreprocessExpectation(
        id="9bn8_cif",
        fixture_name="structure_9bn8_density",
        expected_filtered_atoms=3260,
        expected_hydrogen_atoms=0,
    ),
)


ALL_MISMATCH_CASE_IDS: tuple[str, ...] = (
    "identity_no_mismatch",
    "from_2yl0",
    "from_5sop",
    "from_6ni6",
    "from_9bn8",
)


def _copy_structure_dict(structure: dict[str, Any]) -> dict[str, Any]:
    """Create a mutation-safe shallow copy of a parsed structure dictionary.

    Parameters
    ----------
    structure
        Parsed structure dictionary.

    Returns
    -------
    dict[str, Any]
        Copy whose ``"asym_unit"`` and ``"metadata"`` entries are independently mutable.
    """
    copied: dict[str, Any] = dict(structure)
    asym_unit = structure.get("asym_unit")
    if asym_unit is not None and hasattr(asym_unit, "copy"):
        copied["asym_unit"] = asym_unit.copy()
    metadata = structure.get("metadata")
    if isinstance(metadata, dict):
        copied["metadata"] = dict(metadata)
    return copied


def _build_real_case(
    *,
    id: str,
    description: str,
    model_structure: dict[str, Any],
    struct_structure: dict[str, Any],
    expected_n_common: int,
) -> MismatchCase:
    """Construct a real-structure mismatch case from parsed resources.

    Parameters
    ----------
    id
        Case identifier.
    description
        Human-readable description.
    model_structure
        Parsed structure dictionary for the model representation (CIF).
    struct_structure
        Parsed structure dictionary for the structure/input representation (PDB).
    expected_n_common
        Expected common atom count.

    Returns
    -------
    MismatchCase
        Real mismatch case with filtered atom arrays.
    """
    model_raw = cast(AtomArray, ensure_atom_array_stack(model_structure["asym_unit"])[0])
    model_mask = (model_raw.occupancy > 0) & ~np.any(np.isnan(model_raw.coord), axis=-1)
    model_atom_array = cast(AtomArray, model_raw[model_mask])

    struct_raw = cast(AtomArray, ensure_atom_array_stack(struct_structure["asym_unit"])[0])
    struct_mask = (struct_raw.occupancy > 0) & ~np.any(np.isnan(struct_raw.coord), axis=-1)
    struct_atom_array = cast(AtomArray, struct_raw[struct_mask])

    return MismatchCase(
        id=id,
        description=description,
        model_atom_array=model_atom_array,
        struct_atom_array=struct_atom_array,
        expected_n_common=expected_n_common,
        expected_has_mismatch=True,
    )


def _model_space_reward_inputs(n_model: int, batch: int = 1) -> RewardInputs:
    """Construct model-space reward inputs for DPS tests.

    Parameters
    ----------
    n_model
        Number of model atoms.
    batch
        Batch size.

    Returns
    -------
    RewardInputs
        Reward input bundle in model space.
    """
    return RewardInputs(
        elements=torch.ones(batch, n_model),
        b_factors=torch.ones(batch, n_model) * 20.0,
        occupancies=torch.ones(batch, n_model),
        input_coords=torch.randn(batch, n_model, 3),
    )


def _preprocess(
    wrapper: MismatchCaseWrapper,
    structure: dict[str, Any],
    ensemble_size: int = 1,
):
    """Run full structure preprocessing through the public pipeline.

    Parameters
    ----------
    wrapper
        Wrapper that provides model atom array through conditioning.
    structure
        Structure dictionary with an ``"asym_unit"`` atom array.
    ensemble_size
        Number of ensemble members.

    Returns
    -------
    tuple
        ``(processed_structure, features)``.
    """
    features = wrapper.featurize(structure)
    coords = wrapper.initialize_from_prior(batch_size=ensemble_size, features=features)
    processed = process_structure_to_trajectory_input(
        structure=structure,
        coords_from_prior=coords,
        features=features,
        ensemble_size=ensemble_size,
    )
    return processed, features


# TODO: this should probably go elsewhere
def _rmsd(coords_a: torch.Tensor, coords_b: torch.Tensor) -> torch.Tensor:
    """Compute RMSD between two coordinate tensors.

    Parameters
    ----------
    coords_a
        Coordinates with shape ``(..., atoms, 3)``.
    coords_b
        Coordinates with shape ``(..., atoms, 3)``.

    Returns
    -------
    torch.Tensor
        Scalar RMSD.
    """
    return torch.sqrt(torch.mean((coords_a - coords_b) ** 2))


def _model_reference_from_case(case: MismatchCase, reconciler: AtomReconciler) -> torch.Tensor:
    """Project structure coordinates into model space for a case.

    Parameters
    ----------
    case
        Mismatch case.
    reconciler
        Reconciler built from ``case.model_atom_array`` and ``case.struct_atom_array``.

    Returns
    -------
    torch.Tensor
        Model-space reference coordinates with shape ``(1, n_model, 3)``.
    """
    struct_coords = torch.as_tensor(case.struct_atom_array.coord, dtype=torch.float32).unsqueeze(0)
    model_template = torch.as_tensor(case.model_atom_array.coord, dtype=torch.float32).unsqueeze(0)
    return reconciler.struct_to_model(struct_coords, model_template)


@pytest.fixture(scope="session")
def mismatch_case_catalog(
    structure_2yl0: dict,
    structure_2yl0_density: dict,
    structure_5sop: dict,
    structure_5sop_density: dict,
    structure_6ni6: dict,
    structure_6ni6_density: dict,
    structure_9bn8: dict,
    structure_9bn8_density: dict,
) -> dict[str, MismatchCase]:
    """Build full mismatch case catalog.

    Returns
    -------
    dict[str, MismatchCase]
        Mapping from case ID to case definition.
    """
    catalog: dict[str, MismatchCase] = {}

    no_mismatch_atom_array = build_test_atom_array(
        chain_ids=["A"] * 5,
        res_ids=[1, 2, 3, 4, 5],
        atom_names=["N", "CA", "C", "O", "CB"],
        coords=np.column_stack(
            [
                np.arange(5, dtype=np.float32),
                0.25 * np.arange(5, dtype=np.float32) ** 2 + 0.5,
                np.sin(np.arange(5, dtype=np.float32)),
            ]
        ),
    )
    catalog["identity_no_mismatch"] = MismatchCase(
        id="identity_no_mismatch",
        description="Model and structure are already identical.",
        model_atom_array=no_mismatch_atom_array.copy(),
        struct_atom_array=no_mismatch_atom_array.copy(),
        expected_n_common=5,
        expected_has_mismatch=False,
    )

    catalog["from_2yl0"] = _build_real_case(
        id="from_2yl0",
        description="Real pair from 2YL0 density-input CIF versus full PDB.",
        model_structure=structure_2yl0_density,
        struct_structure=structure_2yl0,
        expected_n_common=677,
    )
    catalog["from_5sop"] = _build_real_case(
        id="from_5sop",
        description="Real pair from 5SOP refinement.",
        model_structure=structure_5sop_density,
        struct_structure=structure_5sop,
        expected_n_common=1264,
    )
    catalog["from_6ni6"] = _build_real_case(
        id="from_6ni6",
        description="Real pair from 6NI6 where we take chain A + drop ligand.",
        model_structure=structure_6ni6_density,
        struct_structure=structure_6ni6,
        expected_n_common=1678,
    )
    catalog["from_9bn8"] = _build_real_case(
        id="from_9bn8",
        description="Real pair from 9BN8 where model atoms are fully covered.",
        model_structure=structure_9bn8_density,
        struct_structure=structure_9bn8,
        expected_n_common=3260,
    )

    missing = set(ALL_MISMATCH_CASE_IDS) - set(catalog)
    if missing:
        raise RuntimeError(f"Missing mismatch catalog entries: {sorted(missing)}")

    return catalog


@pytest.fixture(params=ALL_MISMATCH_CASE_IDS, ids=lambda case_id: case_id)
def mismatch_case(
    request: pytest.FixtureRequest,
    mismatch_case_catalog: dict[str, MismatchCase],
) -> MismatchCase:
    """Parametrized mismatch case fixture covering all case tiers.

    Parameters
    ----------
    request
        Pytest request object.
    mismatch_case_catalog
        Full mismatch case catalog.

    Returns
    -------
    MismatchCase
        Copy of selected case.
    """
    case_id = request.param
    return mismatch_case_catalog[case_id].clone()


@pytest.fixture(params=STRUCTURE_PREPROCESS_EXPECTATIONS, ids=lambda exp: exp.id)
def structure_preprocess_case(request: pytest.FixtureRequest) -> StructurePreprocessExpectation:
    """Parametrized preprocessing expectation fixture."""
    return request.param


class TestRealStructurePreprocessing:
    """Real-data preprocessing and reconciliation tests."""

    def test_preprocessing_filters_nan_and_zero_occ(
        self,
        request: pytest.FixtureRequest,
        structure_preprocess_case: StructurePreprocessExpectation,
    ):
        """Filtered arrays contain only finite coordinates and positive occupancy."""
        structure = request.getfixturevalue(structure_preprocess_case.fixture_name)
        raw_atom_array = cast(AtomArray, ensure_atom_array_stack(structure["asym_unit"])[0])
        mask = (raw_atom_array.occupancy > 0) & ~np.any(np.isnan(raw_atom_array.coord), axis=-1)
        filtered_atom_array = cast(AtomArray, raw_atom_array[mask])

        filtered_coords = cast(np.ndarray, filtered_atom_array.coord)
        filtered_occupancy = cast(np.ndarray, filtered_atom_array.occupancy)

        assert len(filtered_atom_array) <= len(raw_atom_array)
        assert np.isfinite(filtered_coords).all()
        assert np.all(filtered_occupancy > 0)

    def test_preprocessing_preserves_valid_atoms(
        self,
        request: pytest.FixtureRequest,
        structure_preprocess_case: StructurePreprocessExpectation,
    ):
        """Filtered atom counts match known."""
        structure = request.getfixturevalue(structure_preprocess_case.fixture_name)
        raw = cast(AtomArray, ensure_atom_array_stack(structure["asym_unit"])[0])
        mask = (raw.occupancy > 0) & ~np.any(np.isnan(raw.coord), axis=-1)
        filtered_atom_array = cast(AtomArray, raw[mask])

        expected = structure_preprocess_case.expected_filtered_atoms
        expected -= structure_preprocess_case.expected_hydrogen_atoms

        assert len(filtered_atom_array) == expected


class TestReconcilerConstruction:
    """Coordinate identity invariants for reconciler construction."""

    def test_common_count_matches_expected(self, mismatch_case: MismatchCase):
        """Common atom count matches the case ground truth."""
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )
        assert reconciler.n_common == mismatch_case.expected_n_common

    def test_mismatch_flag_correct(self, mismatch_case: MismatchCase):
        """Mismatch flag matches case expectation."""
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )
        assert reconciler.has_mismatch == mismatch_case.expected_has_mismatch

    def test_indices_within_bounds(self, mismatch_case: MismatchCase):
        """Model and structure index arrays stay inside valid bounds."""
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )

        assert torch.all(reconciler.model_indices >= 0)
        assert torch.all(reconciler.model_indices < reconciler.n_model)
        assert torch.all(reconciler.struct_indices >= 0)
        assert torch.all(reconciler.struct_indices < reconciler.n_struct)

    def test_common_atoms_have_matching_identity(self, mismatch_case: MismatchCase):
        """Every reconciled common pair has matching normalized atom identity."""
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )

        model_ids = make_normalized_atom_id(mismatch_case.model_atom_array)
        struct_ids = make_normalized_atom_id(mismatch_case.struct_atom_array)

        np.testing.assert_array_equal(
            model_ids[reconciler.model_indices.detach().cpu().numpy()],
            struct_ids[reconciler.struct_indices.detach().cpu().numpy()],
        )


class TestCoordinateProjection:
    """Coordinate projection invariants for struct_to_model mapping."""

    def test_common_atoms_roundtrip(self, mismatch_case: MismatchCase):
        """Common structure coordinates copy exactly into model indices."""
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )

        struct_coords = torch.arange(
            mismatch_case.n_struct * 3,
            dtype=torch.float32,
        ).reshape(1, mismatch_case.n_struct, 3)
        template = torch.full((1, mismatch_case.n_model, 3), -7.0)

        projected = reconciler.struct_to_model(struct_coords, template)

        torch.testing.assert_close(
            projected[0, reconciler.model_indices],
            struct_coords[0, reconciler.struct_indices],
        )

    def test_non_common_atoms_retain_template(self, mismatch_case: MismatchCase):
        """Model only atoms preserve template coordinates after projection."""
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )

        struct_coords = torch.randn(1, mismatch_case.n_struct, 3)
        template = torch.full((1, mismatch_case.n_model, 3), 13.5)
        projected = reconciler.struct_to_model(struct_coords, template)

        common_mask = torch.zeros(mismatch_case.n_model, dtype=torch.bool)
        common_mask[reconciler.model_indices] = True

        if torch.any(~common_mask):
            torch.testing.assert_close(
                projected[0, ~common_mask],
                template[0, ~common_mask],
            )

    def test_projection_is_differentiable(self, mismatch_case: MismatchCase):
        """Projection keeps a valid autograd path from outputs to struct coordinates."""
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )

        struct_coords = torch.randn(2, mismatch_case.n_struct, 3, requires_grad=True)
        template = torch.randn(2, mismatch_case.n_model, 3)
        projected = reconciler.struct_to_model(struct_coords, template)

        grad = torch.autograd.grad(projected.sum(), struct_coords)[0]
        assert grad is not None
        assert torch.isfinite(grad).all()

    def test_batch_dims_propagate(self, mismatch_case: MismatchCase):
        """Projection supports arbitrary leading batch dimensions."""
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )

        struct_coords = torch.randn(3, 2, mismatch_case.n_struct, 3)
        template = torch.zeros(3, 2, mismatch_case.n_model, 3)
        projected = reconciler.struct_to_model(struct_coords, template)

        assert projected.shape == (3, 2, mismatch_case.n_model, 3)


class TestAlignment:
    """Alignment behavior across the mismatch case catalog."""

    def test_known_rotation_recovery(self, mismatch_case: MismatchCase):
        """Known rigid transforms are recovered with near zero RMSD on common atoms."""
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )
        reference = _model_reference_from_case(mismatch_case, reconciler)

        theta = np.deg2rad(35.0)
        rotation = torch.tensor(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=reference.dtype,
        )
        translation = torch.tensor([4.0, -2.0, 3.5], dtype=reference.dtype)

        transformed = reference @ rotation.T + translation
        aligned, _ = reconciler.align(transformed, reference)

        common = reconciler.model_indices
        rmsd = _rmsd(aligned[..., common, :], reference[..., common, :])
        assert rmsd.item() < 1e-4

    def test_non_common_atoms_transform_rigidly(self, mismatch_case: MismatchCase):
        """Returned alignment matches applying the recovered rigid transform to all atoms."""
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )
        reference = _model_reference_from_case(mismatch_case, reconciler)

        theta = np.deg2rad(20.0)
        rotation = torch.tensor(
            [
                [np.cos(theta), 0.0, np.sin(theta)],
                [0.0, 1.0, 0.0],
                [-np.sin(theta), 0.0, np.cos(theta)],
            ],
            dtype=reference.dtype,
        )
        translation = torch.tensor([1.5, -0.75, 2.0], dtype=reference.dtype)

        transformed = reference @ rotation.T + translation
        aligned, transform = reconciler.align(transformed, reference)

        expected = apply_forward_transform(transformed, transform, rotation_only=False)
        torch.testing.assert_close(aligned, torch.as_tensor(expected), atol=1e-6, rtol=1e-6)

    def test_gradient_flow_through_alignment(self, mismatch_case: MismatchCase):
        """Alignment preserves gradients when allow_gradients=True."""
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )
        reference = _model_reference_from_case(mismatch_case, reconciler)

        model_coords = reference.clone().detach().requires_grad_(True)
        aligned, _ = reconciler.align(model_coords, reference, allow_gradients=True)

        grad = torch.autograd.grad(aligned.sum(), model_coords)[0]
        assert grad is not None
        assert torch.isfinite(grad).all()


class TestPreprocessingPipeline:
    """Pipeline level mismatch tests using :class:`MismatchCaseWrapper`."""

    def test_output_coords_model_space(self, mismatch_case: MismatchCase):
        """Processed input coordinates always use model atom count."""
        wrapper = MismatchCaseWrapper(mismatch_case)
        structure = {
            "asym_unit": mismatch_case.struct_atom_array.copy(),
            "metadata": {"id": mismatch_case.id},
        }
        processed, _ = _preprocess(wrapper, structure)

        assert processed.input_coords.shape[-2] == mismatch_case.n_model

    def test_common_atom_coords_match_struct(self, mismatch_case: MismatchCase):
        """Common model slots hold structure coordinates after preprocessing."""
        wrapper = MismatchCaseWrapper(mismatch_case)
        structure = {
            "asym_unit": mismatch_case.struct_atom_array.copy(),
            "metadata": {"id": mismatch_case.id},
        }
        processed, _ = _preprocess(wrapper, structure)
        reconciler = processed.reconciler

        atom_coords = cast(np.ndarray, processed.atom_array.coord)
        torch.testing.assert_close(
            processed.input_coords[0, reconciler.model_indices],
            torch.as_tensor(
                atom_coords[reconciler.struct_indices.detach().cpu().numpy()],
                dtype=processed.input_coords.dtype,
            ),
        )

    def test_reward_inputs_atom_count(self, mismatch_case: MismatchCase):
        """Reward input tensors all have model atom count."""
        wrapper = MismatchCaseWrapper(mismatch_case)
        structure = {
            "asym_unit": mismatch_case.struct_atom_array.copy(),
            "metadata": {"id": mismatch_case.id},
        }
        processed, _ = _preprocess(wrapper, structure)
        reward_inputs = processed.to_reward_inputs(device="cpu")

        n_model = mismatch_case.n_model
        assert reward_inputs.elements.shape[-1] == n_model
        assert reward_inputs.b_factors.shape[-1] == n_model
        assert reward_inputs.occupancies.shape[-1] == n_model
        assert reward_inputs.input_coords.shape[-2] == n_model

    def test_b_factor_override(self, mismatch_case: MismatchCase):
        """Structure B-factors override model template values on common atoms."""
        struct_atom_array = mismatch_case.struct_atom_array.copy()
        struct_b_factors = np.linspace(5.0, 35.0, mismatch_case.n_struct, dtype=np.float32)
        struct_atom_array.set_annotation("b_factor", struct_b_factors)

        case_with_updated_struct = MismatchCase(
            id=mismatch_case.id,
            description=mismatch_case.description,
            model_atom_array=mismatch_case.model_atom_array.copy(),
            struct_atom_array=struct_atom_array,
            expected_n_common=mismatch_case.expected_n_common,
            expected_has_mismatch=mismatch_case.expected_has_mismatch,
        )
        wrapper = MismatchCaseWrapper(case_with_updated_struct)
        structure = {
            "asym_unit": struct_atom_array.copy(),
            "metadata": {"id": mismatch_case.id},
        }

        processed, _ = _preprocess(wrapper, structure)
        reward_inputs = processed.to_reward_inputs(device="cpu")
        reconciler = processed.reconciler

        expected_common_b_factors = torch.as_tensor(
            struct_b_factors[reconciler.struct_indices.detach().cpu().numpy()],
            dtype=reward_inputs.b_factors.dtype,
        )
        torch.testing.assert_close(
            reward_inputs.b_factors[0, reconciler.model_indices],
            expected_common_b_factors,
        )


class TestSamplerStep:
    """EDM sampler mismatch behavior with coordinate level checks."""

    @pytest.fixture
    def sampler(self) -> AF3EDMSampler:
        """Sampler configured for deterministic mismatch tests."""
        config = EDMSamplerConfig(
            augmentation=False,
            align_to_input=True,
            alignment_reverse_diffusion=False,
            scale_guidance_to_diffusion=True,
            device="cpu",
        )
        return AF3EDMSampler(config)

    def _context_with_reference(
        self,
        reconciler: AtomReconciler,
        reference: torch.Tensor,
        *,
        reward=None,
        reward_inputs: RewardInputs | None = None,
    ) -> StepParams:
        """Build deterministic step context with optional reward payload."""
        context = StepParams(
            step_index=0,
            total_steps=5,
            t=torch.tensor([1.0]),
            dt=torch.tensor([-0.1]),
            noise_scale=torch.tensor([0.0]),
        ).with_reconciler(reconciler, reference)

        if reward is not None and reward_inputs is not None:
            context = context.with_reward(reward, reward_inputs)
        return context

    def test_output_preserves_model_space(
        self,
        mismatch_case: MismatchCase,
        sampler: AF3EDMSampler,
    ):
        """Sampler outputs keep model space shape for mismatch and identity cases."""
        wrapper = MismatchCaseWrapper(mismatch_case)
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )
        reference = _model_reference_from_case(mismatch_case, reconciler)

        features = wrapper.featurize({"asym_unit": mismatch_case.struct_atom_array.copy()})
        state = torch.randn(2, mismatch_case.n_model, 3)
        context = self._context_with_reference(reconciler, reference)

        output = sampler.step(state, wrapper, context, features=features)
        assert output.state.shape == state.shape

    def test_alignment_reduces_rmsd(self, mismatch_case: MismatchCase, sampler: AF3EDMSampler):
        """Alignment lowers RMSD on common atoms versus an unaligned sampler run."""
        wrapper = MismatchCaseWrapper(mismatch_case)
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )
        reference = _model_reference_from_case(mismatch_case, reconciler)

        features = wrapper.featurize({"asym_unit": mismatch_case.struct_atom_array.copy()})
        state = torch.randn(1, mismatch_case.n_model, 3)
        context = self._context_with_reference(reconciler, reference)

        config_no_align = EDMSamplerConfig(
            augmentation=False,
            align_to_input=False,
            alignment_reverse_diffusion=False,
            scale_guidance_to_diffusion=True,
            device="cpu",
        )
        sampler_no_align = AF3EDMSampler(config_no_align)

        torch.manual_seed(42)
        output_aligned = sampler.step(state.clone(), wrapper, context, features=features)

        torch.manual_seed(42)
        output_unaligned = sampler_no_align.step(state.clone(), wrapper, context, features=features)

        assert output_aligned.denoised is not None
        assert output_unaligned.denoised is not None

        common = reconciler.model_indices
        rmsd_aligned = _rmsd(output_aligned.denoised[..., common, :], reference[..., common, :])
        rmsd_unaligned = _rmsd(output_unaligned.denoised[..., common, :], reference[..., common, :])

        assert rmsd_aligned.item() < rmsd_unaligned.item()

    def test_dps_gradient_flows_to_common_atoms(
        self,
        mismatch_case: MismatchCase,
        sampler: AF3EDMSampler,
        mock_gradient_reward,
    ):
        """Guided step changes common atom coordinates relative to unguided baseline."""
        wrapper = MismatchCaseWrapper(mismatch_case)
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )
        reference = _model_reference_from_case(mismatch_case, reconciler)

        features = wrapper.featurize({"asym_unit": mismatch_case.struct_atom_array.copy()})
        state = torch.randn(1, mismatch_case.n_model, 3)
        reward_inputs = _model_space_reward_inputs(mismatch_case.n_model)
        context = self._context_with_reference(
            reconciler,
            reference,
            reward=mock_gradient_reward,
            reward_inputs=reward_inputs,
        )

        torch.manual_seed(7)
        baseline = sampler.step(state.clone(), wrapper, context, features=features)

        torch.manual_seed(7)
        guided = sampler.step(
            state.clone(),
            wrapper,
            context,
            scaler=DataSpaceDPSScaler(step_size=0.1),
            features=features,
        )

        assert guided.loss is not None
        assert torch.isfinite(torch.as_tensor(guided.loss)).all()

        delta = guided.state - baseline.state
        common_shift = delta[0, reconciler.model_indices].abs().sum()
        assert common_shift.item() > 0.0

    def test_denoised_is_model_space(self, mismatch_case: MismatchCase, sampler: AF3EDMSampler):
        """Denoised output tensor stays in model atom space."""
        wrapper = MismatchCaseWrapper(mismatch_case)
        reconciler = AtomReconciler.from_arrays(
            mismatch_case.model_atom_array,
            mismatch_case.struct_atom_array,
        )
        reference = _model_reference_from_case(mismatch_case, reconciler)

        features = wrapper.featurize({"asym_unit": mismatch_case.struct_atom_array.copy()})
        state = torch.randn(1, mismatch_case.n_model, 3)
        context = self._context_with_reference(reconciler, reference)

        output = sampler.step(state, wrapper, context, features=features)

        assert output.denoised is not None
        assert output.denoised.shape == (1, mismatch_case.n_model, 3)


class TestTrajectoryScalers:
    """End-to-end trajectory scalers across mismatch case catalog."""

    def _run_scaler(
        self,
        case: MismatchCase,
        scaler_type: str,
        reward,
        device: str = "cpu",
    ) -> Any:
        """Execute a trajectory scaler run for one case and return GuidanceOutput."""
        wrapper = MismatchCaseWrapper(case, device=torch.device(device))
        structure = {
            "asym_unit": case.struct_atom_array.copy(),
            "metadata": {"id": case.id},
        }
        config = EDMSamplerConfig(augmentation=False, align_to_input=True, device=device)
        sampler = AF3EDMSampler(config)
        step_scaler = DataSpaceDPSScaler(step_size=0.01)

        if scaler_type == "pure_guidance":
            return PureGuidance(ensemble_size=1, num_steps=2, guidance_t_start=0.0).sample(
                structure=structure,
                model=wrapper,
                sampler=sampler,
                step_scaler=step_scaler,
                reward=reward,
            )

        if scaler_type == "fk_steering":
            return FKSteering(
                ensemble_size=1,
                num_steps=2,
                resampling_interval=1,
                fk_lambda=1.0,
                guidance_t_start=0.0,
            ).sample(
                structure=structure,
                model=wrapper,
                sampler=sampler,
                step_scaler=step_scaler,
                reward=reward,
                num_particles=2,
            )

        raise ValueError(f"Unknown scaler_type: {scaler_type}")

    @pytest.mark.parametrize("scaler_type", ["pure_guidance", "fk_steering"])
    def test_output_model_space(
        self,
        mismatch_case: MismatchCase,
        scaler_type: str,
        mock_gradient_reward,
    ):
        """Final states from trajectory scalers stay in model atom space."""
        result = self._run_scaler(mismatch_case, scaler_type, mock_gradient_reward)
        assert result.final_state.shape[-2] == mismatch_case.n_model

    @pytest.mark.parametrize("scaler_type", ["pure_guidance", "fk_steering"])
    def test_reconciliation_metadata(
        self,
        mismatch_case: MismatchCase,
        scaler_type: str,
        mock_gradient_reward,
    ):
        """Metadata carries the canonical reconciliation inputs only for mismatches."""
        result = self._run_scaler(mismatch_case, scaler_type, mock_gradient_reward)
        metadata = result.metadata or {}

        if mismatch_case.expected_has_mismatch:
            model_atom_array = metadata.get("model_atom_array")
            struct_atom_array = metadata.get("struct_atom_array")
            reconciler = metadata.get("reconciler")
            assert model_atom_array is not None
            assert struct_atom_array is not None
            assert isinstance(reconciler, AtomReconciler)
            assert len(model_atom_array) == reconciler.n_model
            assert len(struct_atom_array) == reconciler.n_struct
            assert reconciler.model_indices.device.type == "cpu"
            assert reconciler.struct_indices.device.type == "cpu"
        else:
            assert metadata.get("model_atom_array") is None
            assert metadata.get("struct_atom_array") is None
            assert metadata.get("reconciler") is None

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("scaler_type", ["pure_guidance", "fk_steering"])
    def test_reconciliation_metadata_remains_on_cpu_during_cuda_sampling(
        self,
        mismatch_case_catalog: dict[str, MismatchCase],
        scaler_type: str,
        mock_gradient_reward,
    ):
        """Metadata retains the canonical CPU reconciler during CUDA sampling."""
        case = mismatch_case_catalog["from_2yl0"]

        result = self._run_scaler(case, scaler_type, mock_gradient_reward, device="cuda")

        metadata = result.metadata or {}
        reconciler = metadata.get("reconciler")
        assert isinstance(reconciler, AtomReconciler)
        assert result.final_state.device.type == "cuda"
        assert reconciler.model_indices.device.type == "cpu"
        assert reconciler.struct_indices.device.type == "cpu"

    @pytest.mark.parametrize("scaler_type", ["pure_guidance", "fk_steering"])
    def test_trajectory_shapes_consistent(
        self,
        mismatch_case: MismatchCase,
        scaler_type: str,
        mock_gradient_reward,
    ):
        """All trajectory tensors preserve the model atom dimension."""
        result = self._run_scaler(mismatch_case, scaler_type, mock_gradient_reward)
        assert result.trajectory is not None

        for step_tensor in result.trajectory:
            assert step_tensor.shape[-2] == mismatch_case.n_model

    def test_fk_trajectory_4d(self, mismatch_case: MismatchCase, mock_gradient_reward):
        """FK steering trajectories remain 4D (particles, ensemble, atoms, xyz)."""
        result = self._run_scaler(mismatch_case, "fk_steering", mock_gradient_reward)
        assert result.trajectory is not None

        for step_tensor in result.trajectory:
            assert step_tensor.ndim == 4


# ---------------------------------------------------------------------------
# Shape guard and save behavior
# ---------------------------------------------------------------------------


class TestShapeGuard:
    """Step scalers reject mismatched state/reward_inputs shapes."""

    def test_data_space_dps_rejects_wrong_atom_count(
        self,
        mock_trajectory_context,
        mock_gradient_reward,
    ):
        """DataSpaceDPS raises when state and reward atom counts differ."""
        n_state, n_reward = 8, 5
        context = mock_trajectory_context.with_reward(
            mock_gradient_reward,
            _model_space_reward_inputs(n_reward),
        )
        with pytest.raises(ValueError, match="State atom count.*!=.*reward_inputs atom count"):
            DataSpaceDPSScaler(step_size=0.1).scale(
                torch.randn(1, n_state, 3, requires_grad=True),
                context,
            )

    def test_noise_space_dps_rejects_wrong_atom_count(
        self,
        mock_trajectory_context,
        mock_gradient_reward,
    ):
        """NoiseSpaceDPS raises when state and reward atom counts differ."""
        n_state, n_reward = 8, 5
        x_t = torch.randn(1, n_state, 3, requires_grad=True)
        context = mock_trajectory_context.with_reward(
            mock_gradient_reward,
            _model_space_reward_inputs(n_reward),
        ).with_metadata({"x_t": x_t})

        with pytest.raises(ValueError, match="State atom count.*!=.*reward_inputs atom count"):
            NoiseSpaceDPSScaler(step_size=0.1).scale(torch.randn(1, n_state, 3), context)


class TestSave:
    """save_everything works with model space atom templates."""

    def test_save_with_model_template(self, tmp_path: Path):
        """Saving succeeds with a model template larger than structure atom count."""
        n_struct, n_model = 5, 8
        refined = {"asym_unit": build_test_atom_array(n_atoms=n_struct)}
        model_atom_array = build_test_atom_array(n_atoms=n_model, with_occupancy=False)
        struct_atom_array = refined["asym_unit"]
        reconciler = AtomReconciler.from_arrays(model_atom_array, struct_atom_array)

        args = GuidanceConfig(
            protein="1l63",
            structure=Path("dummy"),
            density=Path("dummy"),
            model_name="boltz2",
            guidance_type="pure_guidance",
            log_path="dummy",
            output_dir=str(tmp_path),
        )

        save_everything(
            args,
            losses=[0.5, 0.3],
            refined_structure=refined,
            traj_denoised=[],
            traj_next_step=[],
            scaler_type="pure_guidance",
            final_state=torch.randn(1, n_model, 3),
            model_atom_array=model_atom_array,
            struct_atom_array=struct_atom_array,
            reconciler=reconciler,
        )

        assert (tmp_path / "refined.cif").exists()
        assert (tmp_path / "losses.txt").exists()
