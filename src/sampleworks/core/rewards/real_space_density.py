from typing import Any, cast

import numpy as np
import torch
from biotite.structure import AtomArray, AtomArrayStack
from jaxtyping import ArrayLike, Float, Int
from loguru import logger
from sampleworks.core.forward_models.xray.real_space_density import (
    DifferentiableTransformer,
    XMap_torch,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import (
    ATOM_STRUCTURE_FACTORS,
    ELECTRON_SCATTERING_FACTORS,
    ELEMENT_TO_SCATTERING_INDEX,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (
    XMap,
)
from sampleworks.core.rewards.options import RealSpaceDensityOptions
from sampleworks.core.rewards.registry import RewardBuildContext
from sampleworks.utils.elements import elements_to_scattering_indices
from sampleworks.utils.torch_utils import try_gpu


def setup_scattering_params(em_mode: bool, device: torch.device) -> torch.Tensor:
    """Set up atomic scattering parameters for density calculation.

    The returned tensor is indexed by the values in
    :data:`ELEMENT_TO_SCATTERING_INDEX`

    Any element without a tabulated scattering factor is left as zeros and logged
    as it indicates an unrecognised element in the input

    Parameters
    ----------
    em_mode
        If ``True``, use electron scattering factors for cryo-EM. If
        ``False``, use X-ray scattering factors.
    device
        PyTorch device that should own the returned lookup tensor.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_indices, n_coeffs, 2)`` containing scattering
        coefficients.
    """
    structure_factors = ELECTRON_SCATTERING_FACTORS if em_mode else ATOM_STRUCTURE_FACTORS
    n_coeffs = len(structure_factors["C"][0])
    n_elements = max(ELEMENT_TO_SCATTERING_INDEX.values()) + 1
    scattering_tensor = torch.zeros((n_elements, n_coeffs, 2), dtype=torch.float32, device=device)

    for element, idx in ELEMENT_TO_SCATTERING_INDEX.items():
        if element not in structure_factors or element == "?":
            logger.warning(
                f"Element '{element}' (scattering index {idx}) has no tabulated "
                "scattering factors and will contribute zero density. This indicates "
                "the scattering factor table is incomplete for this element."
            )
            continue
        factor_tensor = torch.tensor(
            structure_factors[element], dtype=torch.float32, device=device
        ).T
        scattering_tensor[idx] = factor_tensor

    return scattering_tensor


def extract_density_inputs_from_atomarray(
    atom_array_or_stack: AtomArray | AtomArrayStack, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract and prepare atomic data for density calculation.

    Filters out atoms with invalid coordinates or zero occupancy, converts
    element names to atomic numbers, and handles NaN B-factors.

    For :class:`~biotite.structure.AtomArrayStack` inputs, coordinates from all
    models are used (one per batch entry) with shared element types,
    B-factors, and occupancy annotations. Note that this does mean that the occupancy
    values may sum to greater than 1 - we log a warning in this case but keep the provided
    occupancies unchanged.

    Parameters
    ----------
    atom_array_or_stack : AtomArray | AtomArrayStack
        Structure to extract data from. May be a single
        :class:`~biotite.structure.AtomArray` or a multi-model
        :class:`~biotite.structure.AtomArrayStack`.
    device : torch.device-like
        PyTorch device to place tensors on

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple of (coordinates, elements, b_factors, occupancies) as PyTorch
        tensors. For a single ``AtomArray`` the shapes are
        ``(1, n_atoms, 3)``, ``(1, n_atoms)``, ``(1, n_atoms)``,
        ``(1, n_atoms)``. For an ``AtomArrayStack`` with *M* models the batch
        dimension is *M* instead of 1.
    """

    is_stack = isinstance(atom_array_or_stack, AtomArrayStack)

    coords = cast(np.ndarray[Any, np.dtype[np.float64]], atom_array_or_stack.coord)
    occupancy = cast(np.ndarray[Any, np.dtype[np.float64]], atom_array_or_stack.occupancy)
    b_factor = cast(np.ndarray[Any, np.dtype[np.float64]], atom_array_or_stack.b_factor)
    elements = cast(np.ndarray[Any, np.dtype[np.str_]], atom_array_or_stack.element)

    if is_stack:
        # coords is (batch, n_atoms, 3), annotations are (n_atoms,)
        n_models = coords.shape[0]
        n_total = coords.shape[1]
        # NOTE: an atom is invalid if ANY model has non-finite coordinates currently
        invalid_coords_mask = ~np.isfinite(coords).all(axis=-1).all(axis=0)
    else:
        n_models = 1
        n_total = len(coords)
        invalid_coords_mask = ~np.isfinite(coords).all(axis=-1)

    zero_occ_mask = occupancy <= 0
    valid_mask = ~invalid_coords_mask & ~zero_occ_mask

    n_invalid_coords = int(invalid_coords_mask.sum())
    n_zero_occ = int((zero_occ_mask & ~invalid_coords_mask).sum())
    n_valid = int(valid_mask.sum())

    if n_invalid_coords > 0 or n_zero_occ > 0:
        logger.info(
            f"Filtered {n_invalid_coords + n_zero_occ} atoms: "
            f"{n_invalid_coords} with invalid coords, {n_zero_occ} with zero occupancy "
            f"({n_valid}/{n_total} atoms remaining)"
        )

    # TODO can we use [..., valid_mask] here?
    if is_stack:
        valid_coords = coords[:, valid_mask]
    else:
        valid_coords = coords[valid_mask]

    valid_elements = elements[valid_mask]
    valid_b_factor = b_factor[valid_mask]
    valid_occupancy = occupancy[valid_mask]

    if is_stack:
        if not np.all(valid_occupancy * n_models <= 1.0):
            logger.warning(
                f"AtomArrayStack with {n_models} models: occupancy values sum to > 1 for "
                f"some atoms (max sum {valid_occupancy.max() * n_models:.4f}). This may lead to "
                "higher density values than expected under "
                f"uniform model weighting (expected {1 / n_models:.4f}). Keeping "
                "provided occupancies unchanged."
            )

    coords_tensor = torch.from_numpy(valid_coords.copy()).to(device, dtype=torch.float32)
    element_indices = elements_to_scattering_indices(valid_elements)
    elements_tensor = torch.tensor(element_indices, device=device, dtype=torch.long)
    b_factors_tensor = torch.from_numpy(valid_b_factor.copy()).to(device, dtype=torch.float32)

    nan_b_factor_mask = torch.isnan(b_factors_tensor)
    num_nan_b_factors = int(nan_b_factor_mask.sum().item())
    if num_nan_b_factors > 0:
        logger.warning(
            f"{num_nan_b_factors} atoms have NaN B-factors; assigning default B-factor of 20.0"
        )
    b_factors_tensor = torch.where(
        nan_b_factor_mask,
        torch.tensor(20.0, device=device),
        b_factors_tensor,
    )
    occupancies_tensor = torch.from_numpy(valid_occupancy.copy()).to(device, dtype=torch.float32)

    if is_stack:
        # coords_tensor is already (n_models, n_valid, 3)
        # Expand shared annotations to (n_models, n_valid)
        elements_tensor = elements_tensor.unsqueeze(0).expand(n_models, -1)
        b_factors_tensor = b_factors_tensor.unsqueeze(0).expand(n_models, -1)
        occupancies_tensor = occupancies_tensor.unsqueeze(0).expand(n_models, -1)
    else:
        coords_tensor = coords_tensor.unsqueeze(0)
        elements_tensor = elements_tensor.unsqueeze(0)
        b_factors_tensor = b_factors_tensor.unsqueeze(0)
        occupancies_tensor = occupancies_tensor.unsqueeze(0)

    return coords_tensor, elements_tensor, b_factors_tensor, occupancies_tensor


class RealSpaceRewardFunction:
    def __init__(
        self,
        xmap: XMap,
        scattering_params: torch.Tensor,
        selection: ArrayLike | torch.Tensor,
        em: bool = False,
        loss_order: int = 2,
        device: torch.device | None = None,
    ):
        """Hardcoded reward function for now, L1 or L2 reward for fitting real space
        electron density.

        TODO: decide whether these should take in structure or take in
        coords, bfactors, etc. separately. Leaning towards separate, as then
        the functions will be pure and we can do grad w.r.t. input."""

        if device is None:
            device = try_gpu()

        self.transformer = DifferentiableTransformer(
            xmap=XMap_torch(xmap, device=device),
            scattering_params=scattering_params,
            em=em,
            device=device,
        )

        # TODO: selection doesn't do anything right now
        self.selection = (
            selection
            if torch.is_tensor(selection)
            else torch.tensor(selection).to(device=device, dtype=torch.bool)
        )
        self.device = device
        if loss_order == 1:
            self.loss = torch.nn.L1Loss()
        elif loss_order == 2:
            self.loss = torch.nn.MSELoss()
        else:
            raise ValueError("Invalid loss_order, must be 1 or 2")

    def precompute_unique_combinations(
        self,
        elements: Int[torch.Tensor, "batch n_atoms"],
        b_factors: Float[torch.Tensor, "batch n_atoms"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute unique (element, b_factor) combinations for vmap compatibility.

        This allows torch.unique to be called outside of vmap contexts, avoiding
        the dynamic shapes.

        Parameters
        ----------
        elements: torch.Tensor
            Atomic elements, shape [batch n_atoms]
        b_factors: torch.Tensor
            Per-atom B-factors, shape [batch n_atoms]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            unique_combinations: Unique (element, b_factor) pairs
            inverse_indices: Indices to reconstruct original from unique
        """
        device = self.device
        elements_flat = elements.reshape(-1)
        b_factors_flat = b_factors.reshape(-1)
        combined = torch.stack([elements_flat, b_factors_flat], dim=1)
        unique_combinations, inverse_indices = torch.unique(combined, dim=0, return_inverse=True)
        return unique_combinations.to(device), inverse_indices.to(device)

    def structure_to_reward_input(
        self, structure: dict
    ) -> dict[str, Float[torch.Tensor, "..."] | Int[torch.Tensor, "..."]]:
        coordinates, elements, b_factors, occupancies = extract_density_inputs_from_atomarray(
            structure["asym_unit"], self.device
        )

        return {
            "coordinates": coordinates,
            "elements": elements,
            "b_factors": b_factors,
            "occupancies": occupancies,
        }

    def __call__(
        self,
        coordinates: Float[torch.Tensor, "batch n_atoms 3"],
        elements: Int[torch.Tensor, "batch n_atoms"],
        b_factors: Float[torch.Tensor, "batch n_atoms"],
        occupancies: Float[torch.Tensor, "batch n_atoms"],
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, ""]:
        """Pure function for computing reward. Call .backward() on this to get gradients
        w.r.t. input.

        Parameters
        ----------
        coordinates: Float[torch.Tensor, "batch n_atoms 3"]
            Atomic coordinates
        elements: Int[torch.Tensor, "batch n_atoms"]
            Atomic elements
        b_factors: Float[torch.Tensor, "batch n_atoms"]
            Per-atom B-factor
        occupancies: Float[torch.Tensor, "batch n_atoms"]
            Per-atom occupancies
        unique_combinations: torch.Tensor | None, optional
            Pre-computed unique (element, b_factor) pairs for vmap compatibility
        inverse_indices: torch.Tensor | None, optional
            Pre-computed inverse indices for vmap compatibility

        Returns
        -------
        torch.Tensor
            Reward value
        """
        density: torch.Tensor = self.transformer(
            coordinates=coordinates,
            elements=elements,
            b_factors=b_factors,
            occupancies=occupancies,
            unique_combinations=unique_combinations,
            inverse_indices=inverse_indices,
        ).sum(0)  # sum over batch dimension

        return self.loss(density, self.transformer.xmap.array)


def build_real_space_density_reward(
    options: RealSpaceDensityOptions, context: RewardBuildContext
) -> RealSpaceRewardFunction:
    """Build the real-space density reward from its configured options.

    Parameters
    ----------
    options
        Density reward options: map path, resolution, loss order, and whether to
        use electron scattering factors.
    context
        Run-level inputs; the parsed input structure and the target device.

    Returns
    -------
    RealSpaceRewardFunction
        Reward scoring the model against the given map.

    Raises
    ------
    ValueError
        If the map or the resolution is missing.
    """
    if options.density is None:
        raise ValueError(
            "The real_space_density reward needs a density map. Pass --density, or set "
            "reward_options.density in a --reward-config file."
        )
    if options.resolution is None:
        raise ValueError(
            "The real_space_density reward needs a map resolution. Pass --resolution, or "
            "set reward_options.resolution in a --reward-config file."
        )

    device = torch.device(context.device) if isinstance(context.device, str) else context.device

    logger.debug(f"Loading density map from {options.density}")
    xmap = XMap.fromfile(options.density, resolution=options.resolution)

    logger.debug("Setting up scattering parameters")
    atom_array = context.structure["asym_unit"]
    scattering_params = setup_scattering_params(em_mode=options.em, device=device)

    selection_mask = atom_array.occupancy > 0
    logger.info(f"Selected {selection_mask.sum()} atoms with occupancy > 0")

    return RealSpaceRewardFunction(
        xmap,
        scattering_params,
        selection_mask,
        em=options.em,
        loss_order=options.loss_order,
        device=device,
    )
