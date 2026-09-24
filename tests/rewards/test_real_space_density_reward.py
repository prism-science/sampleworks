"""Tests specific to the real-space density reward function.

Reward-agnostic contract tests (interface, relative correlation, coordinate gradients,
batch/edge handling) have been extracted to ``test_reward_function_contract.py``, where
they run against every reward function. What remains here is specific to
``RealSpaceRewardFunction`` (``sampleworks.core.rewards.real_space_density``) or to the
real-space density forward model:

1. Scattering-parameter setup / element normalization (forward model).
2. ``precompute_unique_combinations`` / vmap support (``PrecomputableRewardFunctionProtocol``).
3. ``structure_to_reward_input`` (not implemented by SFC). (Batch=1 occupancy gradients are
   now reward-agnostic and live in ``test_reward_function_contract.py``.)
4. The single-atom case (SFC fixes topology at ``prepare()``).
"""

from functools import partial
from typing import cast

import einx
import numpy as np
import pytest
import torch
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import (
    ATOM_STRUCTURE_FACTORS,
    ELEMENT_TO_ATOMIC_NUM,
)
from sampleworks.core.rewards.real_space_density import (
    extract_density_inputs_from_atomarray,
    RealSpaceRewardFunction,
    setup_scattering_params,
)
from sampleworks.utils.elements import it92_coefficients


class TestSetupScatteringParams:
    """Test structure-independent scattering parameter lookup construction."""

    def test_setup_scattering_params_covers_supported_elements(self, device):
        """Every key in ATOM_STRUCTURE_FACTORS must have a nonzero row in the
        scattering tensor, and its exact coefficients must match the table."""
        from sampleworks.core.rewards.real_space_density import ELEMENT_TO_SCATTERING_INDEX

        params = setup_scattering_params(em_mode=False, device=device)

        assert params.shape[0] == max(ELEMENT_TO_SCATTERING_INDEX.values()) + 1

        for element, coeffs in ATOM_STRUCTURE_FACTORS.items():
            idx = ELEMENT_TO_SCATTERING_INDEX[element]
            expected = torch.tensor(coeffs, dtype=torch.float32, device=device).T
            torch.testing.assert_close(
                params[idx],
                expected,
                msg=lambda m: f"Mismatch for '{element}': {m}",
            )

    def test_unknown_placeholder_row_is_zeros(self, device):
        """The '?' placeholder occupies atomic number 0 and has no scattering factors,
        so its row in the lookup table must be all zeros."""
        params = setup_scattering_params(em_mode=False, device=device)
        assert torch.all(params[ELEMENT_TO_ATOMIC_NUM["?"]] == 0)

    def test_pdb_allcaps_elements_resolve_to_correct_atomic_numbers(
        self, atom_array_with_nan_coords, device
    ):
        """PDB files store elements in all-caps (e.g. 'NA', 'CA', 'SE').
        normalize_element must convert these to standard title-case symbols so they
        map to the correct atomic number and carry nonzero scattering factors.
        """
        params = setup_scattering_params(em_mode=False, device=device)
        aa = atom_array_with_nan_coords.copy()
        aa.element = np.array(["NA", "C", "CA", "C", "SE"])
        _, elements, _, _ = extract_density_inputs_from_atomarray(aa, device)
        expected = torch.tensor(
            [ELEMENT_TO_ATOMIC_NUM[e] for e in ["Na", "Ca", "Se"]],
            dtype=torch.long,
            device=device,
        )
        assert torch.equal(elements.squeeze(0), expected)
        assert all(params[idx].sum().item() != 0 for idx in expected)

    def test_ionic_element_gets_proper_density(self, atom_array_with_nan_coords, device):
        """Ionic element symbols such as 'O1-' are present in ATOM_STRUCTURE_FACTORS
        but absent from ELEMENT_TO_ATOMIC_NUM (which only holds neutral-atom symbols).
        We need to detect this and return the proper scattering params."""
        params = setup_scattering_params(em_mode=False, device=device)
        aa = atom_array_with_nan_coords.copy()
        aa.element = np.array(["O1-", "C", "FE2+", "C", "CL1-"])
        _, elements, _, _ = extract_density_inputs_from_atomarray(aa, device)
        surviving_elements = elements.squeeze(0)

        assert all(idx.item() != 0 for idx in surviving_elements)

        # Verify exact scattering factors from the ATOM_STRUCTURE_FACTORS table
        for idx, ionic_key in zip(surviving_elements, ["O1-", "Fe2+", "Cl1-"]):
            expected = torch.tensor(
                ATOM_STRUCTURE_FACTORS[ionic_key],
                dtype=torch.float32,
                device=device,
            ).T
            torch.testing.assert_close(params[idx], expected)

    def test_completely_unknown_element_falls_back_to_zero_density(
        self, atom_array_with_nan_coords, device
    ):
        """An element symbol that is not recognised at all (neither in
        ELEMENT_TO_ATOMIC_NUM nor in the structure factor table) falls back to
        atomic number 0 and contributes zero density."""
        params = setup_scattering_params(em_mode=False, device=device)
        aa = atom_array_with_nan_coords.copy()
        aa.element = np.array(["Xx", "C", "C", "C", "C"])
        _, elements, _, _ = extract_density_inputs_from_atomarray(aa, device)
        assert elements.squeeze(0)[0].item() == 0
        assert params[0].sum().item() == 0

    def test_it92_mode_reproduces_it92_scattering_factors(self, device):
        """The IT92 table is four Gaussians plus a constant, padded into the
        five-plus-constant layout, so the scattering factor f(s) it yields must
        equal the one computed from the IT92 coefficients directly."""
        from sampleworks.core.rewards.real_space_density import ELEMENT_TO_SCATTERING_INDEX

        params = setup_scattering_params(em_mode=False, device=device, it92_mode=True)
        s = torch.linspace(0.0, 0.6, 7, device=device)

        for element, coefficients in it92_coefficients(["H", "C", "N", "O", "S", "Se"]).items():
            row = params[ELEMENT_TO_SCATTERING_INDEX[element]]
            padded = (row[:5, 0:1] * torch.exp(-row[:5, 1:2] * s**2)).sum(0) + row[5, 0]
            a = torch.tensor(coefficients[:4], device=device).unsqueeze(1)
            b = torch.tensor(coefficients[4:8], device=device).unsqueeze(1)
            expected = (a * torch.exp(-b * s**2)).sum(0) + coefficients[8]
            torch.testing.assert_close(
                padded, expected, msg=lambda m: f"Mismatch for '{element}': {m}"
            )

    def test_it92_mode_leaves_untabulated_rows_zero(self, device):
        """IT92 has no entry for qfit's valence-state pseudo-elements, so those
        rows stay zero rather than borrowing a neighbouring element's factors."""
        from sampleworks.core.rewards.real_space_density import ELEMENT_TO_SCATTERING_INDEX

        params = setup_scattering_params(em_mode=False, device=device, it92_mode=True)
        assert params[ELEMENT_TO_SCATTERING_INDEX["Cval"]].sum().item() == 0
        assert params[ELEMENT_TO_SCATTERING_INDEX["C"]].sum().item() != 0

    def test_it92_mode_rejects_em_mode(self, device):
        """IT92 tabulates X-ray scattering only; combining it with electron
        scattering factors is an error, not a silent preference."""
        with pytest.raises(ValueError, match="X-ray only"):
            setup_scattering_params(em_mode=True, device=device, it92_mode=True)


class TestStructureToRewardInput:
    def test_reuses_density_input_preprocessing(self, atom_array_with_nan_coords, device):
        """Filter invalid/zero-occupancy atoms and replace valid NaN B-factors."""
        atom_array = atom_array_with_nan_coords.copy()
        atom_array.b_factor = np.array([10.0, 15.0, np.nan, 15.0, 40.0])
        atom_array.occupancy = np.array([1.0, 1.0, 1.0, 1.0, 0.0])

        reward_function = RealSpaceRewardFunction.__new__(RealSpaceRewardFunction)
        reward_function.device = device
        inputs = reward_function.structure_to_reward_input({"asym_unit": atom_array})
        expected_coordinates, expected_elements, expected_b_factors, expected_occupancies = (
            extract_density_inputs_from_atomarray(atom_array, device)
        )

        torch.testing.assert_close(inputs["coordinates"], expected_coordinates)
        torch.testing.assert_close(inputs["elements"], expected_elements)
        torch.testing.assert_close(inputs["b_factors"], expected_b_factors)
        torch.testing.assert_close(inputs["occupancies"], expected_occupancies)
        assert inputs["coordinates"].device == device
        assert inputs["elements"].device == device
        assert inputs["b_factors"].device == device
        assert inputs["occupancies"].device == device

        assert inputs["coordinates"].shape == (1, 2, 3)
        assert inputs["coordinates"].dtype == torch.float32
        assert inputs["elements"].dtype == torch.long
        assert inputs["b_factors"].dtype == torch.float32
        assert inputs["occupancies"].dtype == torch.float32
        torch.testing.assert_close(inputs["b_factors"], torch.tensor([[10.0, 20.0]], device=device))

    def test_matches_extractor_batching_for_atom_array_stacks(
        self, atom_array_stack_with_nan_coords, device
    ):
        """Keep extractor batching semantics for multi-model inputs."""
        atom_array_stack = atom_array_stack_with_nan_coords.copy()
        atom_array_stack.b_factor = np.array([np.nan, 20.0, 20.0])
        atom_array_stack.occupancy = np.array([0.5, 0.5, 0.0])

        reward_function = RealSpaceRewardFunction.__new__(RealSpaceRewardFunction)
        reward_function.device = device
        inputs = reward_function.structure_to_reward_input({"asym_unit": atom_array_stack})
        expected_coordinates, expected_elements, expected_b_factors, expected_occupancies = (
            extract_density_inputs_from_atomarray(atom_array_stack, device)
        )

        torch.testing.assert_close(inputs["coordinates"], expected_coordinates)
        torch.testing.assert_close(inputs["elements"], expected_elements)
        torch.testing.assert_close(inputs["b_factors"], expected_b_factors)
        torch.testing.assert_close(inputs["occupancies"], expected_occupancies)

        assert inputs["coordinates"].shape == (2, 1, 3)
        torch.testing.assert_close(inputs["b_factors"], torch.full((2, 1), 20.0, device=device))


@pytest.mark.gpu
@pytest.mark.slow
class TestVmapCompatibility:
    """Test vmap functionality for use in FK steering and particle methods."""

    def test_vmap_over_particle_dimension(
        self, reward_function_1vme, test_coordinates_1vme, device
    ):
        """Test vmap over particle dimension as used in FK steering."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        num_particles = 3
        ensemble_size = 3

        coords_batch = einx.rearrange("n c -> p e n c", coords, p=num_particles, e=ensemble_size)
        elements_batch = einx.rearrange("n -> p e n", elements, p=num_particles, e=ensemble_size)
        b_factors_batch = einx.rearrange("n -> p e n", b_factors, p=num_particles, e=ensemble_size)
        occupancies_batch = einx.rearrange(
            "n -> p e n", occupancies, p=num_particles, e=ensemble_size
        )

        unique_combinations, inverse_indices = reward_function_1vme.precompute_unique_combinations(
            elements_batch[0], b_factors_batch[0]
        )

        rf_partial = partial(
            reward_function_1vme,
            unique_combinations=unique_combinations,
            inverse_indices=inverse_indices,
        )

        result = cast(
            torch.Tensor,
            einx.vmap(
                "p [e n c], p [e n], p [e n], p [e n] -> p",
                coords_batch,
                elements_batch,
                b_factors_batch,
                occupancies_batch,
                op=rf_partial,
            ),
        )

        assert result.shape == torch.Size([num_particles])
        assert torch.all(torch.isfinite(result))

    def test_vmap_with_precomputed_combinations(
        self, reward_function_1vme, test_coordinates_1vme, device
    ):
        """Test vmap with pre-computed unique combinations."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        unique_combinations, inverse_indices = reward_function_1vme.precompute_unique_combinations(
            elements, b_factors
        )

        loss_with_precompute = reward_function_1vme(
            coordinates=coords.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
            unique_combinations=unique_combinations,
            inverse_indices=inverse_indices,
        )

        loss_without_precompute = reward_function_1vme(
            coordinates=coords.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        torch.testing.assert_close(loss_with_precompute, loss_without_precompute)

    def test_vmap_output_shape(self, reward_function_1vme, test_coordinates_1vme, device):
        """Test vmap returns correct shape (num_particles,)."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        for num_particles in [1, 3, 5]:
            coords_batch = einx.rearrange("n c -> p e n c", coords, p=num_particles, e=1)
            elements_batch = einx.rearrange("n -> p e n", elements, p=num_particles, e=1)
            b_factors_batch = einx.rearrange("n -> p e n", b_factors, p=num_particles, e=1)
            occupancies_batch = einx.rearrange("n -> p e n", occupancies, p=num_particles, e=1)

            unique_combinations, inverse_indices = (
                reward_function_1vme.precompute_unique_combinations(
                    elements_batch[0, 0],
                    b_factors_batch[0, 0],
                )
            )

            rf_partial = partial(
                reward_function_1vme,
                unique_combinations=unique_combinations,
                inverse_indices=inverse_indices,
            )

            result = einx.vmap(
                "p [e n c], p [e n], p [e n], p [e n] -> p",
                coords_batch,
                elements_batch,
                b_factors_batch,
                occupancies_batch,
                op=rf_partial,
            )

            assert result.shape == torch.Size([num_particles])

    def test_vmap_consistency(self, reward_function_1vme, test_coordinates_1vme, device):
        """Test vmap results match sequential calls."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        num_particles = 3
        coords_batch = einx.rearrange("n c -> p e n c", coords, p=num_particles, e=1)
        elements_batch = einx.rearrange("n -> p e n", elements, p=num_particles, e=1)
        b_factors_batch = einx.rearrange("n -> p e n", b_factors, p=num_particles, e=1)
        occupancies_batch = einx.rearrange("n -> p e n", occupancies, p=num_particles, e=1)

        unique_combinations, inverse_indices = reward_function_1vme.precompute_unique_combinations(
            elements_batch[0, 0],
            b_factors_batch[0, 0],
        )

        rf_partial = partial(
            reward_function_1vme,
            unique_combinations=unique_combinations,
            inverse_indices=inverse_indices,
        )

        result_vmap = einx.vmap(
            "p [e n c], p [e n], p [e n], p [e n] -> p",
            coords_batch,
            elements_batch,
            b_factors_batch,
            occupancies_batch,
            op=rf_partial,
        )

        result_sequential = []
        for i in range(num_particles):
            loss = rf_partial(
                coordinates=coords_batch[i],
                elements=elements_batch[i],
                b_factors=b_factors_batch[i],
                occupancies=occupancies_batch[i],
            )
            result_sequential.append(loss.item())

        result_sequential = torch.tensor(result_sequential, device=result_vmap.device)

        # GPU vmap and sequential loops accumulate floating-point reductions in
        # different orders, yielding abs diffs up to ~1.3e-4 and rel diffs up to
        # ~6.7e-2 (observed on CI with a single A100).
        torch.testing.assert_close(result_vmap, result_sequential, rtol=1e-1, atol=5e-4)


@pytest.mark.gpu
@pytest.mark.slow
class TestEdgeCases:
    """RealSpace-specific edge cases (shared edge cases are in the contract file)."""

    def test_single_atom(self, reward_function_1vme, test_coordinates_1vme, device):
        """Test with just one atom."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor([ELEMENT_TO_ATOMIC_NUM["C"]], device=device, dtype=torch.float32)
        b_factors = torch.tensor([20.0], device=device, dtype=torch.float32)
        occupancies = torch.tensor([1.0], device=device, dtype=torch.float32)
        coords_single = coords[:1]

        loss = reward_function_1vme(
            coordinates=coords_single.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        assert torch.isfinite(loss)

    def test_structure_to_reward_input(self, reward_function_1vme, structure_1vme_density):
        """Test structure_to_reward_input function."""
        inputs = reward_function_1vme.structure_to_reward_input(structure_1vme_density)

        assert "coordinates" in inputs
        assert "elements" in inputs
        assert "b_factors" in inputs
        assert "occupancies" in inputs

        # Check shapes - should be batched [B, N, ...]
        assert inputs["coordinates"].ndim == 3
        assert inputs["coordinates"].shape[0] == 1
        assert inputs["elements"].ndim == 2
        assert inputs["elements"].shape[0] == 1
        assert inputs["b_factors"].ndim == 2
        assert inputs["b_factors"].shape[0] == 1
        assert inputs["occupancies"].ndim == 2
        assert inputs["occupancies"].shape[0] == 1

        loss = reward_function_1vme(**inputs)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0
