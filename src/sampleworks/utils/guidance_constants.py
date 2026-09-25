from __future__ import annotations

from enum import StrEnum


class GuidanceType(StrEnum):
    """
    Enum for guidance/scaler types used in diffusion model guidance.

    References:
    - Feynman-Kaç steering http://arxiv.org/abs/2501.06848
    - Pure guidance (DPS) http://arxiv.org/abs/2209.14687
    """

    FK_STEERING = "fk_steering"
    PURE_GUIDANCE = "pure_guidance"


class StructurePredictor(StrEnum):
    """
    Enum for supported structure prediction models.
    """

    BOLTZ_1 = "boltz1"
    BOLTZ_2 = "boltz2"
    PROTENIX = "protenix"
    RF3 = "rf3"
    PROTPARDELLE = "protpardelle"


class TrajectorySamplers(StrEnum):
    """Enum for all TrajectorySampler implementations."""

    AF3EDM = "af3edm"


class StepScalers(StrEnum):
    """Enum for all StepScalerProtocol implementations."""

    NO_SCALING = "no_scaling"
    DATA_SPACE_DPS = "data_space_dps"
    NOISE_SPACE_DPS = "noise_space_dps"


class TrajectoryScalers(StrEnum):
    """Enum for all TrajectoryScalerProtocol implementations."""

    PURE_GUIDANCE = "pure_guidance"
    FK_STEERING = "fk_steering"


class Rewards(StrEnum):
    """Enum for all RewardFunctionProtocol implementations."""

    REAL_SPACE_DENSITY = "real_space_density"
    STRUCTURE_FACTOR = "structure_factor"


class Boltz2Method(StrEnum):
    """Enum for Boltz2 sampling methods.

    See https://github.com/jwohlwend/boltz/blob/cb04aeccdd480fd4db707f0bbafde538397fa2ac/src/boltz/data/const.py#L440
    """

    MD = "MD"
    X_RAY_DIFFRACTION = "X-RAY DIFFRACTION"
    ELECTRON_MICROSCOPY = "ELECTRON MICROSCOPY"
    SOLUTION_NMR = "SOLUTION NMR"
    SOLID_STATE_NMR = "SOLID-STATE NMR"
    NEUTRON_DIFFRACTION = "NEUTRON DIFFRACTION"
    ELECTRON_CRYSTALLOGRAPHY = "ELECTRON CRYSTALLOGRAPHY"
    FIBER_DIFFRACTION = "FIBER DIFFRACTION"
    POWDER_DIFFRACTION = "POWDER DIFFRACTION"
    INFRARED_SPECTROSCOPY = "INFRARED SPECTROSCOPY"
    FLUORESCENCE_TRANSFER = "FLUORESCENCE TRANSFER"
    EPR = "EPR"
    THEORETICAL_MODEL = "THEORETICAL MODEL"
    SOLUTION_SCATTERING = "SOLUTION SCATTERING"
    OTHER = "OTHER"
    AFDB = "AFDB"
    BOLTZ_1 = "BOLTZ-1"
