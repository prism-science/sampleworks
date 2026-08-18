"""Recovery test harness for diffuse guidance.

The question this exists to answer: does guiding on diffuse scattering pull an
ensemble toward the one that produced the data?

Answering it needs a target the model can actually reach. Generating targets from
a deposited structure does not qualify — the first attempt on 1VME compared a
model with 6613 atoms against a target built from 6462, sharing only 4515, with
sulfur in the model where the target had selenium, because Boltz renders
selenomethionine as methionine. That mismatch is a floor no guidance removes, and
it would be reported as "diffuse guidance does not work".

So the truth ensemble here is built from the **model's own atom array**: featurize
once, take the topology the model actually produces, displace a chosen set of
atoms into a two-state ensemble, and generate the targets from that. Model and
target are then the same chemical object and the only difference left is the one
deliberately introduced.

Two modes:

    generate   build the truth ensemble and its Bragg/diffuse MTZs, and print the
               guidance command to run against them
    evaluate   compare a guided ensemble against the truth ensemble

The metric that matters is not RMSD to the mean. Diffuse constrains the *spread*,
so evaluate reports the per-atom RMSF of both ensembles: whether guidance put the
motion where the data says it is.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import gemmi
import numpy as np
import torch
from atomworks.io.utils.io_utils import load_any
from biotite.structure import stack
from loguru import logger
from sampleworks.eval.structure_utils import process_structure_to_trajectory_input
from sampleworks.synthetic.generate_synthetic_sf_lunus import (
    compute_ensemble_amplitudes,
    dataset_from_amplitudes,
    dataset_from_intensities,
)
from sampleworks.utils.atom_array_utils import save_structure_to_cif
from sampleworks.utils.guidance_script_utils import get_model_and_device
from sampleworks.utils.torch_utils import try_gpu


DEFAULT_B_FACTOR = 20.0


def model_topology_and_coords(
    structure_path: Path, checkpoint: Path, device: torch.device, out_dir: Path
):
    """Featurize once and return the model's own atom array and its coordinates.

    This is the first half of what a trajectory scaler does. The point is
    ``model_atom_array``: the topology the generative model produces, which is
    what the sampled coordinates will be in and therefore what a target has to be
    generated from.

    Returns
    -------
    atom_array
        The model's topology, with B-factors filled in where absent.
    coords
        ``(n_atoms, 3)`` reference coordinates in model space -- the input
        structure, mapped through the reconciler.
    """
    from sampleworks.models.boltz.wrapper import process_structure_for_boltz
    from sampleworks.utils.guidance_script_utils import _load_structure

    # get_model_and_device reads only `.method` off the config for Boltz-2, so a
    # stand-in carries it. Building a real GuidanceConfig would make this harness
    # satisfy every required argument of the guidance CLI -- --resolution among
    # them -- to do nothing but load a checkpoint.
    config = SimpleNamespace(method="X-RAY DIFFRACTION")

    _, model = get_model_and_device(str(device), str(checkpoint), "boltz2", config=config)
    structure = process_structure_for_boltz(_load_structure(structure_path), out_dir=out_dir)
    features = model.featurize(structure)

    prior = torch.as_tensor(model.initialize_from_prior(batch_size=1, features=features))
    processed = process_structure_to_trajectory_input(
        structure=structure, coords_from_prior=prior, features=features, ensemble_size=1
    )

    atom_array = processed.model_atom_array or processed.atom_array
    coords = processed.input_coords[0].detach().cpu().numpy().astype(np.float64)

    # RewardInputs rejects NaN B-factors, and a model atom array may carry none.
    b_factors = np.asarray(getattr(atom_array, "b_factor", None), dtype=np.float64)
    if b_factors.size != len(atom_array) or not np.isfinite(b_factors).all():
        atom_array.set_annotation(
            "b_factor", np.full(len(atom_array), DEFAULT_B_FACTOR, dtype=np.float32)
        )
    if not hasattr(atom_array, "occupancy"):
        atom_array.set_annotation("occupancy", np.ones(len(atom_array), dtype=np.float32))

    logger.info(
        f"Model topology: {len(atom_array)} atoms, elements {sorted(set(atom_array.element))}"
    )
    return atom_array, coords


def build_truth_ensemble(atom_array, coords, residue_range, displacement):
    """Displace one residue range in opposite directions to make a two-state ensemble.

    A collective displacement rather than per-atom noise, deliberately. Noise
    produces diffuse intensity too, but nothing recoverable: there is no
    structure for guidance to find. A loop occupying two positions is the
    simplest thing that is both physically meaningful and unambiguously
    recoverable, and it puts the diffuse signal on a known set of atoms.

    Returns ``(n_configs, n_atoms, 3)`` with the two states, and the boolean mask
    of the atoms that moved.
    """
    first, last = residue_range
    moving = (atom_array.res_id >= first) & (atom_array.res_id <= last)
    if not moving.any():
        raise ValueError(
            f"No atoms in residues {first}-{last}; the model's numbering runs "
            f"{atom_array.res_id.min()}-{atom_array.res_id.max()}."
        )

    # A fixed, arbitrary direction: the axis does not matter, only that both
    # states differ from the mean by a known amount.
    direction = np.array([1.0, 0.0, 0.0])
    offset = np.zeros_like(coords)
    offset[moving] = displacement * direction

    truth = np.stack([coords + offset, coords - offset])
    logger.info(
        f"Truth ensemble: 2 states, {int(moving.sum())} of {len(atom_array)} atoms "
        f"displaced by ±{displacement} A (residues {first}-{last})"
    )
    return truth, moving


def rmsf(coords: np.ndarray) -> np.ndarray:
    """Per-atom root-mean-square fluctuation about the ensemble mean."""
    return np.sqrt(((coords - coords.mean(axis=0)) ** 2).sum(axis=-1).mean(axis=0))


def generate(args: argparse.Namespace) -> None:
    """Build the truth ensemble and its targets from the model's own topology."""
    device = try_gpu() if not args.device else torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    atom_array, coords = model_topology_and_coords(
        args.structure, args.model_checkpoint, device, args.output_dir
    )
    truth, moving = build_truth_ensemble(
        atom_array, coords, (args.first_residue, args.last_residue), args.displacement
    )

    meta = gemmi.read_structure(str(args.structure))
    cell, spacegroup = meta.cell, gemmi.SpaceGroup(meta.spacegroup_hm)

    hkl, mean_f, diffuse = compute_ensemble_amplitudes(
        atom_array, truth, cell, spacegroup, args.resolution, device
    )
    if diffuse.max() <= 0:
        raise RuntimeError(
            "The truth ensemble has no diffuse signal. Check that the displacement "
            "is large enough to exceed the float32 noise floor."
        )

    bragg_path = args.output_dir / "truth_bragg.mtz"
    diffuse_path = args.output_dir / "truth_diffuse.mtz"
    dataset_from_amplitudes(
        hkl, mean_f, cell, spacegroup, test_fraction=0.0, output_path=bragg_path
    )
    dataset_from_intensities(hkl, diffuse, cell, spacegroup, output_path=diffuse_path)

    truth_stack = stack([atom_array.copy() for _ in range(truth.shape[0])])
    truth_stack.coord = truth.astype(np.float32)
    truth_path = args.output_dir / "truth_ensemble.cif"
    save_structure_to_cif(truth_stack, truth_path)

    logger.info(
        f"Truth RMSF: {rmsf(truth)[moving].mean():.3f} A over moved atoms, "
        f"{rmsf(truth)[~moving].mean():.3f} A elsewhere"
    )
    logger.info(f"Wrote {bragg_path}, {diffuse_path}, {truth_path}")

    print("\nRun guidance against this target with:\n")
    print(
        f"  pixi run -e boltz sampleworks-guidance \\\n"
        f"      --model boltz2 --guidance-type pure_guidance --protein recovery \\\n"
        f"      --model-checkpoint {args.model_checkpoint} \\\n"
        f"      --structure {args.structure} \\\n"
        f"      --target-type diffuse \\\n"
        f"      --bragg-target {bragg_path} \\\n"
        f"      --diffuse-target {diffuse_path} \\\n"
        f"      --bragg-weight {args.bragg_weight} \\\n"
        f"      --resolution {args.resolution} --ensemble-size {args.ensemble_size} \\\n"
        f"      --augmentation --align-to-input \\\n"
        f"      --output-dir {args.output_dir / 'guided'}\n"
    )
    print(
        f"Then: python {Path(__file__).name} evaluate "
        f"--truth {truth_path} --guided {args.output_dir / 'guided' / 'refined.cif'}\n"
    )


def evaluate(args: argparse.Namespace) -> None:
    """Compare a guided ensemble's spread against the truth ensemble's.

    RMSD to the mean is the wrong headline number here: diffuse constrains the
    second moment, so what matters is whether the motion ended up on the atoms
    the data says are moving. The reported correlation between the two RMSF
    profiles is that question.
    """
    # load_any, not load_structure_with_altlocs: the latter takes the first model
    # only, which would silently reduce both ensembles to one structure and make
    # the spread comparison -- the entire measurement -- vacuous.
    truth = load_any(args.truth, extra_fields=["occupancy", "b_factor"])
    guided = load_any(args.guided, extra_fields=["occupancy", "b_factor"])
    logger.info(f"Loaded truth {args.truth} and guided {args.guided}")

    truth_coords = np.asarray(truth.coord, dtype=np.float64)
    guided_coords = np.asarray(guided.coord, dtype=np.float64)
    if truth_coords.ndim != 3 or guided_coords.ndim != 3:
        raise ValueError(
            "Both inputs must be multi-model. A single-model file has no spread to "
            "compare, which is the whole measurement."
        )
    if truth_coords.shape[1] != guided_coords.shape[1]:
        raise ValueError(
            f"Atom counts differ: truth {truth_coords.shape[1]}, guided "
            f"{guided_coords.shape[1]}. They must be the same topology."
        )

    truth_rmsf, guided_rmsf = rmsf(truth_coords), rmsf(guided_coords)
    correlation = float(np.corrcoef(truth_rmsf, guided_rmsf)[0, 1])

    print(f"\n  truth RMSF   mean {truth_rmsf.mean():.3f} A, max {truth_rmsf.max():.3f} A")
    print(f"  guided RMSF  mean {guided_rmsf.mean():.3f} A, max {guided_rmsf.max():.3f} A")
    print(f"  RMSF correlation  {correlation:.4f}")
    print(
        "\n  Correlation near 1 means guidance put the motion where the data says.\n"
        "  Near 0 means the ensemble spread is unrelated to the target, whatever\n"
        "  the loss did.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    gen = sub.add_parser("generate", help="Build the truth ensemble and its targets")
    gen.add_argument("--structure", type=Path, required=True)
    gen.add_argument("--model-checkpoint", type=Path, default=Path("/checkpoints/boltz2_conf.ckpt"))
    gen.add_argument("--output-dir", type=Path, default=Path("output/diffuse_recovery"))
    gen.add_argument("--resolution", type=float, default=1.8)
    gen.add_argument(
        "--first-residue", type=int, required=True, help="First residue of the range to displace"
    )
    gen.add_argument("--last-residue", type=int, required=True)
    gen.add_argument(
        "--displacement",
        type=float,
        default=0.5,
        help="Half-separation of the two states, Angstroms",
    )
    gen.add_argument("--ensemble-size", type=int, default=4)
    gen.add_argument(
        "--bragg-weight",
        type=float,
        default=0.0,
        help="Printed into the suggested command; 0 scores diffuse alone",
    )
    gen.add_argument("--device", type=str, default="")
    gen.set_defaults(func=generate)

    ev = sub.add_parser("evaluate", help="Compare a guided ensemble against the truth")
    ev.add_argument("--truth", type=Path, required=True)
    ev.add_argument("--guided", type=Path, required=True)
    ev.set_defaults(func=evaluate)

    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
