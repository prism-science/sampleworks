"""Shared AtomArray builders for tests."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from biotite.structure import array, Atom, AtomArray
from sampleworks.utils.atom_array_utils import (
    BLANK_ALTLOC_IDS,
    load_structure_with_altlocs,
    map_altlocs_to_stack,
    save_structure_to_cif,
)


def build_test_atom_array(
    *,
    n_atoms: int | None = None,
    chain_ids: Sequence[str] | None = None,
    res_ids: Sequence[int] | None = None,
    atom_names: Sequence[str] | None = None,
    coords: np.ndarray | None = None,
    res_name: str = "ALA",
    element: str = "C",
    with_occupancy: bool = True,
    occupancy_value: float = 1.0,
    b_factor_value: float = 20.0,
) -> AtomArray:
    """Build a minimal AtomArray with configurable annotations.

    Parameters
    ----------
    n_atoms
        Number of atoms when explicit annotation lists are not provided.
    chain_ids
        Per-atom chain IDs.
    res_ids
        Per-atom residue IDs.
    atom_names
        Per-atom atom names.
    coords
        Optional coordinate array with shape ``(n_atoms, 3)``.
    res_name
        Residue name assigned to all atoms.
    element
        Element symbol assigned to all atoms.
    with_occupancy
        Whether to attach an occupancy annotation.
    occupancy_value
        Occupancy value used when ``with_occupancy=True``.
    b_factor_value
        B-factor value assigned to all atoms.

    Returns
    -------
    AtomArray
        Configured atom array for tests.
    """
    if chain_ids is not None and res_ids is not None and atom_names is not None:
        if not (len(chain_ids) == len(res_ids) == len(atom_names)):
            raise ValueError("chain_ids, res_ids, and atom_names must have matching lengths")
        n = len(chain_ids)
    elif n_atoms is not None:
        n = n_atoms
        chain_ids = ["A"] * n
        res_ids = list(range(1, n + 1))
        atom_names = [f"C{i}" for i in range(n)]
    else:
        raise ValueError("Provide either n_atoms or all of chain_ids/res_ids/atom_names")

    if coords is None:
        coords = np.zeros((n, 3), dtype=np.float32)
    if coords.shape != (n, 3):
        raise ValueError(f"coords must have shape ({n}, 3), got {coords.shape}")

    atoms = []
    for i, (chain_id, res_id, atom_name) in enumerate(zip(chain_ids, res_ids, atom_names)):
        atoms.append(
            Atom(
                coord=np.asarray(coords[i], dtype=np.float32),
                chain_id=str(chain_id),
                res_id=int(res_id),
                res_name=res_name,
                atom_name=str(atom_name),
                element=element,
            )
        )

    arr = array(atoms)
    if with_occupancy:
        arr.set_annotation("occupancy", np.full(n, occupancy_value, dtype=np.float32))
    arr.set_annotation("b_factor", np.full(n, b_factor_value, dtype=np.float32))
    return arr


def altlocs_to_ensemble_cif(cif_path: Path, output_path: Path) -> Path:
    """Rewrite a single-model altloc structure as a multi-model ensemble reference.

    Each altloc becomes one model. Anchor atoms keep a blank altloc ID and other atoms get
    altloc ``X``, following the layout of ``build_reference_frames``. The ordering of the
    models corresponds to the ordering of altloc IDs in the single-model altloc structure.

    Parameters
    ----------
    cif_path
        Single-model structure with altlocs.
    output_path
        Where to write the multi-model ensemble CIF.

    Returns
    -------
    Path
        The written ensemble CIF.

    Raises
    ------
    ValueError
        If stacking drops any atom, e.g. one present in only some altlocs, since the ensemble
        would then no longer describe the same structure.
    """
    atom_array = load_structure_with_altlocs(cif_path)
    ensemble, removed = map_altlocs_to_stack(atom_array)
    dropped = set(zip(atom_array.chain_id, atom_array.res_id, atom_array.atom_name)) - set(
        zip(ensemble.chain_id, ensemble.res_id, ensemble.atom_name)
    )
    if dropped:
        raise ValueError(f"{cif_path} loses atoms when stacked by altloc: {sorted(dropped)[:10]}")
    # altloc annotation should be consistent across models
    is_anchor = np.isin(removed["altloc_id"][0], list(BLANK_ALTLOC_IDS))
    ensemble.set_annotation("altloc_id", np.where(is_anchor, "", "X"))
    # load_any keeps altloc_id only when the file has an occupancy column; per-model values
    # cannot be stored on a stack, so model 0's stand in
    ensemble.set_annotation("occupancy", removed["occupancy"][0])
    ensemble.set_annotation("b_factor", removed["b_factor"][0])
    return save_structure_to_cif(ensemble, output_path)
