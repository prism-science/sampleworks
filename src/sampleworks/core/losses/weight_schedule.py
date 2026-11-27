"""How much of a loss's weight applies as guidance progresses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeightSchedule:
    """Fraction of a loss's weight in force at each point of the guided steps.

    ``progress`` runs from 0 at the first guided step to 1 at the last (see
    :func:`guidance_progress`). The weight is off before ``start``, full from
    ``end`` on, and rises linearly in between, so ``start == end`` switches it on
    in a single step and ``start=0, end=1`` ramps it over the whole run.

    Scheduling a physical-plausibility prior late lets the data term fit first:
    a bond-geometry penalty applied from the start steers against the density
    while the structure is still coarse, but applied over the last guided steps
    it repairs geometry without costing fit.

    Parameters
    ----------
    start
        Progress at which the weight starts to apply, in [0, 1].
    end
        Progress from which the full weight applies, in [``start``, 1].

    Raises
    ------
    ValueError
        If the bounds are outside [0, 1] or ``end`` precedes ``start``.
    """

    start: float
    end: float

    def __post_init__(self):
        if not 0.0 <= self.start <= self.end <= 1.0:
            raise ValueError(
                f"A weight schedule needs 0 <= start <= end <= 1, got start={self.start}, "
                f"end={self.end}."
            )

    def fraction(self, progress: float) -> float:
        """Fraction of the weight that applies at ``progress``.

        Parameters
        ----------
        progress
            Position in the guided steps, 0 at the first and 1 at the last.

        Returns
        -------
        float
            0 before ``start``, 1 from ``end`` on, linear in between.
        """
        if progress >= self.end:
            return 1.0
        if progress < self.start:
            return 0.0
        return (progress - self.start) / (self.end - self.start)


def guidance_progress(step_index: int, first_guided_step: int, num_steps: int) -> float:
    """Position of a step within the guided steps, 0 at the first and 1 at the last.

    Parameters
    ----------
    step_index
        Index of the current step in the full schedule.
    first_guided_step
        Index of the first step that applies guidance.
    num_steps
        Number of steps in the full schedule; the last guided step is ``num_steps - 1``.

    Returns
    -------
    float
        Progress in [0, 1]. A run with a single guided step reports 1, so every
        scheduled weight is in full force on the last guided step.
    """
    last_guided_step = num_steps - 1
    if last_guided_step <= first_guided_step:
        return 1.0
    return (step_index - first_guided_step) / (last_guided_step - first_guided_step)
