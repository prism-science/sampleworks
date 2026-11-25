"""Pack MolProbity's Top8000 Ramachandran and rotamer contour grids into one ``.npz``.

The grids come from https://github.com/rlabduke/reference_data (``Top8000/``, CC BY 4.0;
Williams et al., Protein Sci. 27, 293-315, 2018). Each ``.data`` file lists bin centers and
percentile-contour values; bins that are absent are zero. The packed file stores, per table,
the dense grid and the lower bound and bin width of every axis, which is what
:mod:`sampleworks.core.losses.molprobity` interpolates.

Usage::

    git clone --depth 1 https://github.com/rlabduke/reference_data.git
    python scripts/build_molprobity_tables.py reference_data \
        src/sampleworks/data/molprobity_top8000.npz
"""

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np


RAMACHANDRAN_TABLES = {
    "general": "rama8000-general-noGPIVpreP.data",
    "glycine": "rama8000-gly-sym.data",
    "cis_proline": "rama8000-cispro.data",
    "trans_proline": "rama8000-transpro.data",
    "pre_proline": "rama8000-prepro-noGP.data",
    "isoleucine_valine": "rama8000-ileval-nopreP.data",
}

ROTAMER_TABLES = {
    "ARG": "rota8000-arg.data",
    "ASN": "rota8000-asn.data",
    "ASP": "rota8000-asp.data",
    "CYS": "rota8000-cys.data",
    "GLN": "rota8000-gln.data",
    "GLU": "rota8000-glu.data",
    "HIS": "rota8000-his.data",
    "ILE": "rota8000-ile.data",
    "LEU": "rota8000-leu.data",
    "LYS": "rota8000-lys.data",
    "MET": "rota8000-met.data",
    "PHE_TYR": "rota8000-phetyr.data",
    "PRO": "rota8000-pro.data",
    "SER": "rota8000-ser.data",
    "THR": "rota8000-thr.data",
    "TRP": "rota8000-trp.data",
    "VAL": "rota8000-val.data",
}

_AXIS = re.compile(r"^#\s+x\d+:\s+(\S+)\s+(\S+)\s+(\d+)\s+(true|false)")


def read_contour_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one Kinemage-style n-dimensional contour table.

    Parameters
    ----------
    path
        A ``rama8000-*.data`` or ``rota8000-*.data`` file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Dense grid of shape [n_bins_1, ..., n_bins_k] (float32), and the lower bound and bin
        width of each axis in degrees, shape [k] each.

    Raises
    ------
    ValueError
        If an axis does not wrap, or a listed coordinate is not a bin center.
    """
    axes = []
    rows = []
    for line in path.read_text().splitlines():
        match = _AXIS.match(line)
        if match:
            lower, upper, n_bins, wrapping = match.groups()
            if wrapping != "true":
                raise ValueError(f"{path.name}: expected every axis to wrap, got {line!r}.")
            axes.append((float(lower), float(upper), int(n_bins)))
        elif line and not line.startswith("#"):
            rows.append([float(value) for value in line.split()])

    lower = np.array([axis[0] for axis in axes])
    width = np.array([(axis[1] - axis[0]) / axis[2] for axis in axes])
    shape = tuple(axis[2] for axis in axes)
    table = np.array(rows)  # [n_listed, k + 1]: bin-center coordinates, then value
    bins = (table[:, :-1] - lower) / width - 0.5
    if not np.allclose(bins, np.round(bins), atol=1e-6):
        raise ValueError(f"{path.name}: coordinates are not bin centers.")

    grid = np.zeros(shape, dtype=np.float32)
    grid[tuple(np.round(bins).astype(int).T)] = table[:, -1]
    return grid, lower.astype(np.float32), width.astype(np.float32)


def main():
    """Read the Top8000 grids from a reference_data checkout and write the packed file."""
    parser = argparse.ArgumentParser(
        description="Pack the MolProbity Top8000 contour grids into one .npz."
    )
    parser.add_argument("reference_data", type=Path, help="Checkout of rlabduke/reference_data")
    parser.add_argument("output", type=Path, help="Destination .npz")
    args = parser.parse_args()

    top8000 = args.reference_data / "Top8000"
    arrays = {}
    for prefix, directory, tables in (
        ("rama", "Top8000_ramachandran_pct_contour_grids", RAMACHANDRAN_TABLES),
        ("rota", "Top8000_rotamer_pct_contour_grids", ROTAMER_TABLES),
    ):
        for name, filename in tables.items():
            grid, lower, width = read_contour_grid(top8000 / directory / filename)
            arrays[f"{prefix}/{name}/grid"] = grid
            arrays[f"{prefix}/{name}/lower"] = lower
            arrays[f"{prefix}/{name}/width"] = width

    commit = subprocess.run(
        ["git", "-C", str(args.reference_data), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    arrays["source_commit"] = np.array(commit)
    np.savez_compressed(args.output, **arrays)


if __name__ == "__main__":
    main()
