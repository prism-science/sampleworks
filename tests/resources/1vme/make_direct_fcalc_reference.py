"""Regenerate the cctbx direct-summation structure-factor reference for 1VME.

Cuts a small, fully occupied fragment out of ``1vme_final.cif`` (chain A, residues
14-33, no hydrogens, isotropic B) in the deposited P 1 21 1 cell, and computes its structure factors
by direct summation over atoms and symmetry operations with cctbx and the IT92
form-factor table, the one gemmi, lunus.sf and SFC_Torch use. P 1 21 1 has no special
positions, so occupancy conventions for atoms on symmetry elements do not enter.

cctbx is not a sampleworks dependency, so this runs outside the pixi environments::

    pixi exec -s cctbx-base -s python=3.11 python \
        tests/resources/1vme/make_direct_fcalc_reference.py
"""

from pathlib import Path

import iotbx.pdb


RESOURCE_DIR = Path(__file__).parent
SOURCE_CIF = RESOURCE_DIR / "1vme_final.cif"
FRAGMENT_PDB = RESOURCE_DIR / "1vme_A14-33_fragment.pdb"
REFERENCE_MTZ = RESOURCE_DIR / "1vme_A14-33_fcalc_direct_2.50A.mtz"
FRAGMENT_SELECTION = "chain A and resseq 14:33 and not element H"
D_MIN = 2.5


def main() -> None:
    """Write the fragment model and its direct-summation ``FC``/``PHIFC`` MTZ."""
    source = iotbx.pdb.input(file_name=str(SOURCE_CIF))
    crystal_symmetry = source.crystal_symmetry()
    hierarchy = source.construct_hierarchy()
    selection = hierarchy.atom_selection_cache().selection(FRAGMENT_SELECTION)
    fragment = hierarchy.select(selection)

    occupancies = fragment.atoms().extract_occ()
    if min(occupancies) < 1.0:
        raise ValueError(f"{FRAGMENT_SELECTION!r} selects partially occupied atoms")

    # sampleworks carries one isotropic B per atom, so the deposit's anisotropic ADPs are
    # replaced by their isotropic equivalent (the B column of the deposit).
    for atom in fragment.atoms():
        atom.uij_erase()
    xray_structure = fragment.extract_xray_structure(crystal_symmetry=crystal_symmetry)
    if xray_structure.use_u_aniso().count(True):
        raise ValueError("fragment still carries anisotropic ADPs")
    xray_structure.scattering_type_registry(table="it1992")
    f_calc = xray_structure.structure_factors(d_min=D_MIN, algorithm="direct").f_calc()

    fragment.write_pdb_file(file_name=str(FRAGMENT_PDB), crystal_symmetry=crystal_symmetry)
    f_calc.as_mtz_dataset(column_root_label="FC").mtz_object().write(str(REFERENCE_MTZ))
    print(f"{fragment.atoms_size()} atoms, {f_calc.size()} reflections, {crystal_symmetry}")


if __name__ == "__main__":
    main()
