"""Structure factors from lunus.sf and SFC_Torch against a cctbx direct-summation reference.

The reference is a 184-atom fragment of 1VME in its deposited P 1 21 1 cell, with
``F(hkl)`` to 2.5 A summed directly over atoms and symmetry operations by cctbx with the
IT92 form factors; ``tests/resources/1vme/make_direct_fcalc_reference.py`` regenerates
it. Both forward models use IT92 too, so they should reproduce it to float32 precision;
a wrong phase sign, symmetry operation, cell convention or B-factor would not.
"""

from pathlib import Path
from typing import NamedTuple

import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs
import torch
from biotite.structure import AtomArray
from jaxtyping import Complex, Int
from sampleworks.synthetic.synthetic_utils import atomarray_to_gemmi
from sampleworks.utils.atom_array_utils import load_structure_with_altlocs
from SFC_Torch import SFcalculator
from SFC_Torch.io import PDBParser


REFERENCE_RESOLUTION = 2.5
# Observed agreement is ~1e-4 for both models (float32 accumulation); a symmetry, phase or
# form-factor error is orders of magnitude larger.
MAX_RELATIVE_ERROR = 1e-3


class DirectSummationReference(NamedTuple):
    """The cctbx reference reflections and the crystal they belong to."""

    dataset: rs.DataSet
    hkl: Int[np.ndarray, "n_refl 3"]
    structure_factors: Complex[np.ndarray, " n_refl"]
    unit_cell: gemmi.UnitCell
    space_group: str


def complex_structure_factors(dataset: rs.DataSet) -> Complex[np.ndarray, " n_refl"]:
    """Combine the ``FC`` amplitude and ``PHIFC`` phase (degrees) columns into ``F``."""
    phases = np.deg2rad(dataset["PHIFC"].to_numpy(dtype=np.float64))
    return dataset["FC"].to_numpy(dtype=np.float64) * np.exp(1j * phases)


def relative_error(
    predicted: Complex[np.ndarray, " n_refl"], reference: Complex[np.ndarray, " n_refl"]
) -> float:
    """``sum |F_pred - F_ref| / sum |F_ref|``: zero only if amplitudes and phases agree."""
    return float(np.abs(predicted - reference).sum() / np.abs(reference).sum())


@pytest.fixture(scope="module")
def fragment_dir(resources_dir: Path) -> Path:
    """Directory holding the 1VME fragment and its reference MTZ."""
    return resources_dir / "1vme"


@pytest.fixture(scope="module")
def fragment(fragment_dir: Path) -> AtomArray:
    """The fully occupied, isotropic-B fragment the reference was computed from."""
    return load_structure_with_altlocs(fragment_dir / "1vme_A14-33_fragment.pdb")


@pytest.fixture(scope="module")
def reference(fragment_dir: Path) -> DirectSummationReference:
    """The cctbx direct-summation reflections, in the order cctbx wrote them."""
    dataset = rs.read_mtz(str(fragment_dir / "1vme_A14-33_fcalc_direct_2.50A.mtz"))
    return DirectSummationReference(
        dataset=dataset,
        hkl=dataset.get_hkls(),
        structure_factors=complex_structure_factors(dataset),
        unit_cell=dataset.cell,
        space_group=dataset.spacegroup.hm,
    )


def test_reference_covers_the_resolution_shell(reference: DirectSummationReference):
    """The reference is complete to 2.5 A, so every resolution shell is checked below."""
    d_spacings = reference.unit_cell.calculate_d_array(reference.hkl)
    assert d_spacings.min() == pytest.approx(REFERENCE_RESOLUTION, abs=0.01)
    assert len(reference.hkl) > 30_000


def test_lunus_matches_direct_summation(fragment: AtomArray, reference: DirectSummationReference):
    """The lunus.sf adapter reproduces cctbx's F(hkl), amplitudes and phases."""
    pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")
    from sampleworks.core.forward_models.xray.lunus_sf import build_setup, structure_factors

    setup = build_setup(fragment, reference.unit_cell, reference.space_group, REFERENCE_RESOLUTION)
    coords = torch.as_tensor(fragment.coord, dtype=torch.float32)[None]
    occupancies = torch.as_tensor(fragment.occupancy, dtype=torch.float32)

    predicted = structure_factors(setup, coords, occupancies, torch.as_tensor(reference.hkl))

    error = relative_error(predicted[0].numpy(), reference.structure_factors)
    assert error < MAX_RELATIVE_ERROR, error


def test_sfcalculator_matches_direct_summation(
    fragment: AtomArray, reference: DirectSummationReference
):
    """SFcalculator, built from an atom array as the structure-factor loss builds it,
    reproduces cctbx's F(hkl), amplitudes and phases."""
    gemmi_structure = atomarray_to_gemmi(
        fragment, unit_cell=reference.unit_cell, space_group=reference.space_group
    )
    sfc = SFcalculator(
        pdbmodel=PDBParser(gemmi_structure),
        mtzdata=reference.dataset.copy(),
        set_experiment=False,
        expcolumns=["FC", "PHIFC"],
        device=torch.device("cpu"),
    )
    coords = torch.as_tensor(fragment.coord, dtype=torch.float32)[None]

    predicted = sfc.calc_fprotein_batch(coords, Return=True)[0].detach().numpy()

    # SFcalculator reports reflections in gemmi's reciprocal ASU; map the reference there
    # too (rs shifts the phases of remapped reflections) and align the two by Miller index.
    reference_asu = reference.dataset.hkl_to_asu()
    sfc_order = pd.MultiIndex.from_arrays(sfc.HKL_array.T, names=["H", "K", "L"])
    reference_asu = reference_asu.reindex(sfc_order)
    error = relative_error(predicted, complex_structure_factors(reference_asu))
    assert error < MAX_RELATIVE_ERROR, error
