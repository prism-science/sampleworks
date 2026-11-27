"""Tests for phasing a loss's weight in over the guided steps."""

import pytest
from sampleworks.core.losses.weight_schedule import guidance_progress, WeightSchedule


class TestWeightSchedule:
    def test_weight_is_off_before_start_and_full_from_end(self):
        schedule = WeightSchedule(start=0.6, end=1.0)

        assert schedule.fraction(0.0) == 0.0
        assert schedule.fraction(0.59) == 0.0
        assert schedule.fraction(1.0) == 1.0

    def test_weight_rises_linearly_between_start_and_end(self):
        schedule = WeightSchedule(start=0.6, end=1.0)

        assert schedule.fraction(0.6) == 0.0
        assert schedule.fraction(0.8) == pytest.approx(0.5)
        assert schedule.fraction(0.9) == pytest.approx(0.75)

    def test_equal_bounds_switch_the_weight_on_in_one_step(self):
        schedule = WeightSchedule(start=0.9, end=0.9)

        assert schedule.fraction(0.89) == 0.0
        assert schedule.fraction(0.9) == 1.0

    def test_full_range_ramps_over_the_whole_run(self):
        schedule = WeightSchedule(start=0.0, end=1.0)

        assert [schedule.fraction(p) for p in (0.0, 0.25, 1.0)] == [0.0, 0.25, 1.0]

    @pytest.mark.parametrize(
        ("start", "end"), [(-0.1, 0.5), (0.5, 1.1), (0.8, 0.6)], ids=["below", "above", "reversed"]
    )
    def test_bounds_outside_the_guided_steps_or_reversed_are_rejected(self, start, end):
        with pytest.raises(ValueError, match="0 <= start <= end <= 1"):
            WeightSchedule(start=start, end=end)


class TestGuidanceProgress:
    def test_runs_from_zero_at_the_first_guided_step_to_one_at_the_last(self):
        progress = [guidance_progress(i, first_guided_step=120, num_steps=200) for i in (120, 199)]

        assert progress == [0.0, 1.0]

    def test_is_linear_in_the_step_index(self):
        assert guidance_progress(150, first_guided_step=100, num_steps=201) == pytest.approx(0.5)

    def test_a_single_guided_step_is_the_last_one(self):
        """Every scheduled weight must be in full force on the last guided step."""
        assert guidance_progress(9, first_guided_step=9, num_steps=10) == 1.0
