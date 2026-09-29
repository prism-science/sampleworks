"""Convert a saved final_state.pt (from run_langevin_test.py) into a viewable CIF.

Reconstructs the model atom-array template the same way PureGuidance.sample()
does internally (without re-running the diffusion sampler), then writes the
saved coordinates into it and saves as CIF for visual inspection in
PyMOL/ChimeraX.

Usage
-----
    pixi run -e protenix python scripts/convert_final_state_to_cif.py \
        output/langevin_sweep/csg/final_state.pt output/langevin_sweep/csg/refined.cif
    pixi run -e protenix python scripts/convert_final_state_to_cif.py \
        output/langevin_sweep/factor0.4/final_state.pt output/langevin_sweep/factor0.4/refined.cif
"""

import sys

import torch
from biotite.structure import stack
from biotite.structure.io.pdbx import CIFFile, set_structure

from sampleworks.eval.structure_utils import process_structure_to_trajectory_input
from sampleworks.utils.guidance_script_utils import (
    get_model_and_device,
    get_reward_function_and_structure,
)
from sampleworks.utils.torch_utils import try_gpu


STRUCTURE = "tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY = "tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
RESOLUTION = 1.8


def main() -> None:
    final_state_path, out_cif_path = sys.argv[1], sys.argv[2]
    final_state = torch.load(final_state_path, map_location="cpu")
    ensemble_size = final_state.shape[0]

    device = try_gpu()
    _, model = get_model_and_device(
        device_str=str(device), model_checkpoint_path=None, model_type="protenix"
    )
    _, structure = get_reward_function_and_structure(
        density=DENSITY, device=device, em=False, loss_order=2,
        resolution=RESOLUTION, structure_path=STRUCTURE,
    )

    features = model.featurize(structure)
    prior_coords = torch.as_tensor(
        model.initialize_from_prior(batch_size=ensemble_size, features=features)
    )
    processed_structure = process_structure_to_trajectory_input(
        structure=structure, coords_from_prior=prior_coords, features=features,
        ensemble_size=ensemble_size,
    )
    template = processed_structure.reward_atom_array

    ensemble_array = stack([template.copy() for _ in range(ensemble_size)])
    ensemble_array.coord = final_state.numpy()

    cif_file = CIFFile()
    set_structure(cif_file, ensemble_array)
    cif_file.write(out_cif_path)
    print(f"Wrote {out_cif_path}")


if __name__ == "__main__":
    main()
