"""Black-box tests for reconciled CIF output."""

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from biotite.structure.io.pdbx.cif import CIFFile
from sampleworks.utils.atom_array_utils import load_structure_with_altlocs
from sampleworks.utils.atom_reconciler import AtomReconciler
from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import relabel_model_residues, save_everything

from tests.utils.atom_array_builders import build_test_atom_array


def _guidance_config(output_dir: Path, guidance_type: str = "pure_guidance") -> GuidanceConfig:
    """Build a minimal guidance configuration for save tests.

    Parameters
    ----------
    output_dir
        Directory for serialized outputs.
    guidance_type
        Trajectory scaler identifier.

    Returns
    -------
    GuidanceConfig
        Save-compatible guidance configuration.
    """
    return GuidanceConfig(
        # Not a deposited structure id, so saving never fetches a reference CIF from RCSB.
        protein="example_run",  # placeholder field
        structure=Path("input.cif"),
        density=Path("density.ccp4"),
        model_name="boltz2",  # placeholder field
        guidance_type=guidance_type,
        log_path="test.log",
        output_dir=str(output_dir),
    )


def test_save_restores_input_numbering_and_keeps_model_only_atoms_in_residue(tmp_path: Path):
    """Model-only atoms (e.g. if we add OXT atoms) stay inside their residue even if they aren't at
    the end of the chain."""
    model = build_test_atom_array(
        chain_ids=["A", "A", "A", "A", "A", "A", "B", "B"],
        res_ids=[0, 0, 0, 1, 1, 1, 0, 0],
        atom_names=["N", "CA", "CB", "N", "CA", "OXT", "N", "CA"],
    )
    struct = build_test_atom_array(
        chain_ids=["X", "X", "X", "X", "Y", "Y"],
        res_ids=[365, 365, 366, 366, 20, 20],
        atom_names=["N", "CA", "N", "CA", "N", "CA"],
    )
    reconciler = AtomReconciler.from_arrays(model, struct)
    coords = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)

    save_everything(
        _guidance_config(tmp_path),
        losses=[],
        refined_structure={"asym_unit": struct.copy()},
        traj_denoised=[],
        traj_next_step=[],
        scaler_type="pure_guidance",
        final_state=coords,
        model_atom_array=relabel_model_residues(model, struct, reconciler),
    )

    output_path = tmp_path / "refined.cif"
    output = load_structure_with_altlocs(output_path)
    block = CIFFile.read(str(output_path)).block
    atom_ids = block["atom_site"]["id"].as_array().tolist()
    assert output.chain_id.tolist() == ["X"] * 6 + ["Y"] * 2
    assert output.res_id.tolist() == [365, 365, 365, 366, 366, 366, 20, 20]
    assert output.atom_name.tolist() == ["N", "CA", "CB", "N", "CA", "OXT", "N", "CA"]
    assert len(set(atom_ids)) == len(atom_ids) == len(output)
    np.testing.assert_allclose(output.coord, coords[0].numpy())


def test_save_writes_modeled_met_and_input_b_factors_at_mse_position(tmp_path: Path):
    """A MET modeled for an input MSE is written as MET at the input's numbering.

    The loss scored sulfur, so the output says sulfur; matched atoms carry the input's
    B-factors, which are the ones the loss used.
    """
    model = build_test_atom_array(chain_ids=["A", "A"], res_ids=[0, 0], atom_names=["N", "SD"])
    model.res_name[:] = "MET"
    model.element[:] = ["N", "S"]
    unfiltered = build_test_atom_array(
        chain_ids=["B", "B", "B"],
        res_ids=[13, 13, 13],
        atom_names=["N", "SE", "O"],
    )
    unfiltered.res_name[:] = "MSE"
    unfiltered.element[:] = ["N", "Se", "O"]
    unfiltered.hetero[:] = True
    unfiltered.occupancy[:] = [1.0, 1.0, 0.0]
    unfiltered.b_factor[:] = [31.0, 32.0, 33.0]
    struct = unfiltered[unfiltered.occupancy > 0]
    reconciler = AtomReconciler.from_arrays(model, struct)
    coords = torch.tensor([[[1.0, 0.0, 0.0], [7.0, 8.0, 9.0]]])

    save_everything(
        _guidance_config(tmp_path),
        losses=[],
        refined_structure={"asym_unit": unfiltered},
        traj_denoised=[],
        traj_next_step=[],
        scaler_type="pure_guidance",
        final_state=coords,
        model_atom_array=relabel_model_residues(model, struct, reconciler),
    )

    output = load_structure_with_altlocs(tmp_path / "refined.cif")
    assert output.chain_id.tolist() == ["B", "B"]
    assert output.res_id.tolist() == [13, 13]
    assert output.res_name.tolist() == ["MET", "MET"]
    assert output.atom_name.tolist() == ["N", "SD"]
    assert output.element.tolist() == ["N", "S"]
    np.testing.assert_allclose(output.b_factor, [31.0, 32.0])
    np.testing.assert_allclose(output.coord, coords[0].numpy())


def test_relabel_fills_gaps_per_offset_and_keeps_model_only_chain():
    """Unmatched residues continue from the matched residue before them.
    A model chain absent from the input keeps the model's labels."""
    model = build_test_atom_array(
        chain_ids=["A"] * 12 + ["B"] * 4,
        res_ids=np.repeat([1, 2, 3, 4, 5, 6, 1, 2], 2).tolist(),
        atom_names=["N", "CA"] * 8,
        b_factor_value=20.0,
    )
    struct = build_test_atom_array(
        chain_ids=["X"] * 4,
        res_ids=[11, 11, 20, 20],
        atom_names=["N", "CA"] * 2,
        b_factor_value=50.0,
    )
    # Model residues 2 and 4 in chain A are the matched ones.
    struct.set_annotation("seq_idx", np.array([1, 1, 3, 3]))

    relabeled = relabel_model_residues(model, struct, AtomReconciler.from_arrays(model, struct))

    assert relabeled.chain_id.tolist() == ["X"] * 12 + ["B"] * 4
    assert relabeled.res_id.tolist() == np.repeat([10, 11, 12, 20, 21, 22, 1, 2], 2).tolist()
    assert relabeled.atom_name.tolist() == model.atom_name.tolist()
    np.testing.assert_allclose(
        relabeled.b_factor, np.repeat([20.0, 50.0, 20.0, 50.0, 20.0, 20.0, 20.0, 20.0], 2)
    )


def test_relabel_warns_when_gap_residues_reuse_an_input_label(caplog):
    """A gap residue numbered from its neighbor can end up with an existing label, and
    we warn about it."""
    model = build_test_atom_array(
        chain_ids=["A"] * 3, res_ids=[1, 2, 3], atom_names=["CA"] * 3, b_factor_value=20.0
    )
    struct = build_test_atom_array(
        chain_ids=["X", "X"], res_ids=[11, 12], atom_names=["CA", "CA"], b_factor_value=50.0
    )
    # Model residues 1 and 3 are matched; the gap residue 2 follows residue 1 and takes 12.
    struct.set_annotation("seq_idx", np.array([0, 2]))

    with caplog.at_level(logging.WARNING):
        relabeled = relabel_model_residues(model, struct, AtomReconciler.from_arrays(model, struct))

    assert relabeled.res_id.tolist() == [11, 12, 12]
    assert "duplicate residue labels" in caplog.text


def test_relabel_numbers_residues_the_input_lacks_from_their_neighbors():
    """Termini and gap residues predicted under a sequence override get consecutive numbers."""
    model = build_test_atom_array(
        chain_ids=["A"] * 4, res_ids=[1, 2, 3, 4], atom_names=["CA"] * 4, b_factor_value=20.0
    )
    struct = build_test_atom_array(
        chain_ids=["X", "X"], res_ids=[11, 13], atom_names=["CA", "CA"], b_factor_value=50.0
    )
    # Positions in the override sequence, as apply_sequence_override annotates them.
    struct.set_annotation("seq_idx", np.array([1, 3]))

    relabeled = relabel_model_residues(model, struct, AtomReconciler.from_arrays(model, struct))

    assert relabeled.chain_id.tolist() == ["X"] * 4
    assert relabeled.res_id.tolist() == [10, 11, 12, 13]
    np.testing.assert_allclose(relabeled.b_factor, [20.0, 50.0, 20.0, 50.0])


@pytest.mark.parametrize(
    "guidance_type,trajectory_shape",
    [
        ("pure_guidance", lambda coords: coords),
        ("fk_steering", lambda coords: coords.unsqueeze(0)),
    ],
)
def test_save_maps_each_final_and_trajectory_coordinate_set(
    tmp_path: Path,
    guidance_type: str,
    trajectory_shape,
):
    """Final, denoised, and next-step CIFs each carry their own coordinates and input labels."""
    model = build_test_atom_array(chain_ids=["A", "A"], res_ids=[0, 0], atom_names=["N", "CA"])
    struct = build_test_atom_array(chain_ids=["B", "B"], res_ids=[365, 365], atom_names=["N", "CA"])
    reconciler = AtomReconciler.from_arrays(model, struct)
    final_coords = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    denoised_coords = torch.tensor([[[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    next_coords = torch.tensor([[[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]])

    save_everything(
        _guidance_config(tmp_path, guidance_type),
        losses=[],
        refined_structure={"asym_unit": struct.copy()},
        traj_denoised=[trajectory_shape(denoised_coords)],
        traj_next_step=[trajectory_shape(next_coords)],
        scaler_type=guidance_type,
        final_state=final_coords,
        model_atom_array=relabel_model_residues(model, struct, reconciler),
    )

    paths_and_expected = [
        (tmp_path / "refined.cif", final_coords),
        (tmp_path / "trajectory" / "denoised" / "trajectory_0.cif", denoised_coords),
        (tmp_path / "trajectory" / "next_step" / "trajectory_0.cif", next_coords),
    ]
    for path, expected in paths_and_expected:
        output = load_structure_with_altlocs(path)
        assert output.chain_id.tolist() == ["B", "B"]
        assert output.res_id.tolist() == [365, 365]
        np.testing.assert_allclose(output.coord, expected[0].numpy())
