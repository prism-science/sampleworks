"""Black-box tests for reconciled CIF output."""

from pathlib import Path

import numpy as np
import pytest
import torch
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io.pdbx.cif import CIFFile
from sampleworks.utils.atom_reconciler import AtomReconciler
from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import save_everything

from tests.utils.atom_array_builders import build_test_atom_array


def _load_first_model(path: Path) -> AtomArray:
    """Load the first model from a structure file.

    Parameters
    ----------
    path
        Structure file to load.

    Returns
    -------
    AtomArray
        First model as an atom array.
    """
    result = load_any(path, altloc="all", extra_fields=["occupancy", "b_factor"])
    return result[0] if isinstance(result, AtomArrayStack) else result


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
        protein="test",
        structure=Path("input.cif"),
        density=Path("density.ccp4"),
        model_name="boltz2",
        guidance_type=guidance_type,
        log_path="test.log",
        output_dir=str(output_dir),
    )


def test_save_restores_input_numbering_and_inserts_model_only_atom_in_residue(tmp_path: Path):
    """A generated OXT stays contiguous with its mapped residue before the next chain."""
    model = build_test_atom_array(
        chain_ids=["A", "A", "A", "B", "B"],
        res_ids=[0, 0, 0, 0, 0],
        atom_names=["N", "CA", "OXT", "N", "CA"],
    )
    model.element[:] = ["N", "C", "O", "N", "C"]
    struct = build_test_atom_array(
        chain_ids=["X", "X", "Y", "Y"],
        res_ids=[365, 365, 20, 20],
        atom_names=["N", "CA", "N", "CA"],
    )
    struct.set_annotation("atom_id", np.array([7, 8, 9, 10]))
    reconciler = AtomReconciler.from_arrays(model, struct)
    coords = torch.tensor(
        [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]
    )

    save_everything(
        _guidance_config(tmp_path),
        losses=[],
        refined_structure={"asym_unit": struct.copy()},
        traj_denoised=[],
        traj_next_step=[],
        scaler_type="pure_guidance",
        final_state=coords,
        model_atom_array=model,
        struct_atom_array=struct,
        reconciler=reconciler,
    )

    output_path = tmp_path / "refined.cif"
    output = _load_first_model(output_path)
    cif = CIFFile.read(str(output_path))
    block = cif[next(iter(cif))]
    atom_ids = block["atom_site"]["id"].as_array().tolist()
    assert output.chain_id.tolist() == ["X", "X", "X", "Y", "Y"]
    assert output.res_id.tolist() == [365, 365, 365, 20, 20]
    assert set(output.atom_name[output.chain_id == "X"]) == {"N", "CA", "OXT"}
    assert len(set(atom_ids)) == len(atom_ids) == len(output)
    x_ca = output[(output.chain_id == "X") & (output.atom_name == "CA")]
    np.testing.assert_allclose(x_ca.coord, [[2.0, 0.0, 0.0]])
    np.testing.assert_allclose(output.coord[output.atom_name == "OXT"], [[3.0, 0.0, 0.0]])


def test_save_uses_filtered_count_and_full_input_mse_identity(tmp_path: Path):
    """MSE identity is preserved and filtered input-only coordinates stay off disk."""
    model = build_test_atom_array(
        chain_ids=["A", "A"], res_ids=[0, 0], atom_names=["N", "SD"]
    )
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
        model_atom_array=model,
        struct_atom_array=struct,
        reconciler=reconciler,
    )

    output = _load_first_model(tmp_path / "refined.cif")
    assert len(output) == len(struct) == 2
    selenium = output[output.atom_name == "SE"]
    assert selenium.res_name.tolist() == ["MSE"]
    assert selenium.element.tolist() == ["Se"]
    assert selenium.hetero.tolist() == [True]
    np.testing.assert_allclose(selenium.coord, [[7.0, 8.0, 9.0]])


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
    """Final, denoised, and next-step CIFs each map their own coordinates."""
    model = build_test_atom_array(
        chain_ids=["A", "A"], res_ids=[0, 0], atom_names=["N", "CA"]
    )
    struct = build_test_atom_array(
        chain_ids=["B", "B"], res_ids=[365, 365], atom_names=["N", "CA"]
    )
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
        model_atom_array=model,
        struct_atom_array=struct,
        reconciler=reconciler,
    )

    paths_and_expected = [
        (tmp_path / "refined.cif", 2.0),
        (tmp_path / "trajectory" / "denoised" / "trajectory_0.cif", 4.0),
        (tmp_path / "trajectory" / "next_step" / "trajectory_0.cif", 6.0),
    ]
    for path, expected_x in paths_and_expected:
        output = _load_first_model(path)
        ca = output[(output.chain_id == "B") & (output.res_id == 365) & (output.atom_name == "CA")]
        np.testing.assert_allclose(ca.coord, [[expected_x, 0.0, 0.0]])
