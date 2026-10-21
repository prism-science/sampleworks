"""Shared input builders for loss tests."""

import torch
from sampleworks.utils.elements import elements_to_scattering_indices


def build_scattering_indices(atom_array, device: torch.device) -> torch.Tensor:
    """Map biotite element symbols to scattering-tensor indices (production path).

    Mirrors ``LossInputs.from_atom_array``: uses ``elements_to_scattering_indices``
    (so ionic forms resolve correctly) and ``dtype=torch.long``. Shared by the loss
    tests so they build ``elements`` identically.

    Parameters
    ----------
    atom_array : biotite.structure.AtomArray
        Atoms whose ``element`` annotation is mapped to scattering-tensor indices.
    device : torch.device
        Device the returned tensor is placed on.

    Returns
    -------
    torch.Tensor
        1-D ``torch.long`` tensor of scattering-tensor indices, one per atom, on ``device``.
    """
    return torch.tensor(
        elements_to_scattering_indices(atom_array.element),
        device=device,
        dtype=torch.long,
    )


def build_loss_input_tensors_without_coords(
    atom_array, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``(elements, b_factors, occupancies)`` loss inputs from an atom array.

    Uses the atom array's **actual** per-atom occupancy and B-factor — unlike
    ``LossInputs.from_atom_array``, which overrides occupancy to a uniform
    ``1/ensemble_size``. Loss unit tests score against ground truth (e.g. the 1vme
    0.5/0.5 altloc occupancies), so they need the real values. Elements use
    :func:`build_scattering_indices` (scattering-tensor indices, ``torch.long``).

    Coordinates are intentionally left out: callers already hold them from their
    coordinate fixture (and often perturb them), so only the element/B/occupancy trio is
    genuinely shared across the loss test files.

    Parameters
    ----------
    atom_array : biotite.structure.AtomArray
        Atoms whose ``element``/``b_factor``/``occupancy`` annotations are extracted.
    device : torch.device
        Device the returned tensors are placed on.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(elements [N] long, b_factors [N] float32, occupancies [N] float32)`` on ``device``.
    """
    elements = build_scattering_indices(atom_array, device)
    b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
    occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)
    return elements, b_factors, occupancies
