"""Contract for the re-planning minimum-jerk reach generator.

Two tests carry most of the weight. Undisturbed, the generator must reproduce
the canonical min-jerk curve ``10τ³ - 15τ⁴ + 6τ⁵`` -- re-planning from a point
on your own trajectory is a mathematical no-op, and if that fails the whole
premise is wrong. Disturbed, it must curve back toward the target rather than
snapping to the new bearing, which is the property that replaces a hand-tuned
blend weight.
"""

from __future__ import annotations

import math

import pytest

from ezmsg.kinematics import min_jerk
from ezmsg.kinematics.min_jerk import ReplanSeed

PEAK_SPEED = 300.0
DISTANCE = 300.0
NOMINAL_T = 1.875 * DISTANCE / PEAK_SPEED  # 1.875 s


def _canonical_position(t: float, T: float, start: float, end: float) -> float:
    """Closed-form min-jerk position at time ``t`` for reach ``start→end``."""
    if t <= 0.0:
        return start
    if t >= T:
        return end
    tau = t / T
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    return start + (end - start) * s


def _plan(
    *,
    start_xy=(0.0, 0.0),
    target_xy=(DISTANCE, 0.0),
    peak_speed=PEAK_SPEED,
    t0=0.0,
    **kwargs,
) -> min_jerk.MinJerkReach:
    state = min_jerk.MinJerkReach()
    min_jerk.begin_reach(
        state,
        t0,
        start_xy[0],
        start_xy[1],
        target_xy[0],
        target_xy[1],
        peak_speed,
        **kwargs,
    )
    return state


def _run(
    state: min_jerk.MinJerkReach,
    *,
    dt: float = 1.0 / 100.0,
    duration: float = 3.0,
    start_xy=(0.0, 0.0),
    perturb=None,
):
    """Drive the effector with the generator's own output, forward-Euler.

    ``perturb(step_index, x, y) -> (x, y)`` injects a disturbance, standing in
    for whatever else is pushing the effector in a shared-control setting.
    """
    x, y = start_xy
    trace = [(state.t0, x, y, 0.0, 0.0)]
    for i in range(int(duration / dt)):
        t = state.t0 + (i + 1) * dt
        vx, vy = min_jerk.step(state, t, x, y, movement_allowed=True)
        x += vx * dt
        y += vy * dt
        if perturb is not None:
            x, y = perturb(i, x, y)
        trace.append((t, x, y, vx, vy))
    return trace


class TestQuinticSolver:
    def test_it_hits_every_boundary_condition(self):
        p0, v0, a0, pf, T = 10.0, -35.0, 12.0, 400.0, 1.4
        c3, c4, c5 = min_jerk._quintic_to_rest(p0, v0, a0, pf, T)

        def pos(t):
            return p0 + v0 * t + 0.5 * a0 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5

        v_end, a_end = min_jerk._eval_quintic(v0, a0, c3, c4, c5, T)
        assert pos(0.0) == pytest.approx(p0)
        assert pos(T) == pytest.approx(pf)
        assert v_end == pytest.approx(0.0, abs=1e-9)
        assert a_end == pytest.approx(0.0, abs=1e-9)
        v_start, a_start = min_jerk._eval_quintic(v0, a0, c3, c4, c5, 0.0)
        assert v_start == pytest.approx(v0)
        assert a_start == pytest.approx(a0)

    def test_from_rest_it_is_the_canonical_min_jerk_polynomial(self):
        T, D = 1.875, 300.0
        c3, c4, c5 = min_jerk._quintic_to_rest(0.0, 0.0, 0.0, D, T)
        assert c3 == pytest.approx(10.0 * D / T**3)
        assert c4 == pytest.approx(-15.0 * D / T**4)
        assert c5 == pytest.approx(6.0 * D / T**5)


class TestPlanning:
    def test_duration_is_set_so_the_bell_peaks_at_peak_speed(self):
        state = _plan()
        assert state.D == pytest.approx(DISTANCE)
        assert state.T == pytest.approx(NOMINAL_T)
        assert (state.D / state.T) * 1.875 == pytest.approx(PEAK_SPEED)
        assert state.active is True

    def test_a_longer_reach_takes_longer_rather_than_going_faster(self):
        near = _plan(target_xy=(100.0, 0.0))
        far = _plan(target_xy=(400.0, 0.0))
        assert far.T == pytest.approx(4.0 * near.T)

    def test_a_zero_distance_reach_is_inactive_rather_than_an_error(self):
        state = _plan(start_xy=(50.0, 50.0), target_xy=(50.0, 50.0))
        assert state.active is False
        assert min_jerk.step(state, 1.0, 50.0, 50.0, movement_allowed=True) == (0.0, 0.0)

    def test_a_non_positive_peak_speed_is_inactive_rather_than_an_error(self):
        assert _plan(peak_speed=0.0).active is False

    def test_a_reach_starts_from_rest(self):
        state = _plan()
        assert (state.cmd_vx, state.cmd_vy) == (0.0, 0.0)
        assert (state.cmd_ax, state.cmd_ay) == (0.0, 0.0)


class TestGating:
    def test_no_velocity_before_a_reach_is_planned(self):
        state = min_jerk.MinJerkReach()
        assert min_jerk.step(state, 1.0, 0.0, 0.0, movement_allowed=True) == (0.0, 0.0)

    def test_no_velocity_when_movement_disallowed(self):
        state = _plan()
        assert min_jerk.step(state, 0.2, 0.0, 0.0, movement_allowed=False) == (0.0, 0.0)

    def test_being_gated_off_commands_rest_rather_than_merely_muting(self):
        # The distinction matters on resume: a held-over velocity would be
        # re-planned from as though the effector had been moving all along.
        state = _plan()
        min_jerk.step(state, 0.2, 30.0, 0.0, movement_allowed=True)
        assert state.cmd_vx != 0.0
        min_jerk.step(state, 0.3, 30.0, 0.0, movement_allowed=False)
        assert (state.cmd_vx, state.cmd_vy) == (0.0, 0.0)
        assert (state.cmd_ax, state.cmd_ay) == (0.0, 0.0)

    def test_a_long_pause_does_not_produce_a_lurch_on_resume(self):
        # t_prev tracks through the pause, so the first step back is one control
        # interval, not the whole pause.
        state = _plan(ballistic_duration=0.0)
        min_jerk.step(state, 0.01, 0.0, 0.0, movement_allowed=True)
        for i in range(100):  # 1 s paused
            min_jerk.step(state, 0.02 + i * 0.01, 5.0, 0.0, movement_allowed=False)
        vx, vy = min_jerk.step(state, 1.03, 5.0, 0.0, movement_allowed=True)
        assert math.hypot(vx, vy) < PEAK_SPEED

    def test_reset_stops_the_reach(self):
        state = _plan()
        min_jerk.reset(state)
        assert state.active is False
        assert min_jerk.step(state, 0.2, 0.0, 0.0, movement_allowed=True) == (0.0, 0.0)

    def test_a_timestamp_before_the_reach_yields_nothing(self):
        state = _plan(t0=100.0)
        assert min_jerk.step(state, 99.5, 0.0, 0.0, movement_allowed=True) == (0.0, 0.0)


class TestBallisticPhase:
    def test_it_commands_the_planned_velocity_at_the_elapsed_time(self):
        state = _plan()
        for elapsed in (0.05, 0.2, 0.4):
            vx, vy = min_jerk.step(state, elapsed, 0.0, 0.0, movement_allowed=True)
            tau = elapsed / state.T
            expected = (state.D / state.T) * 30.0 * tau**2 * (1.0 - tau) ** 2
            assert math.hypot(vx, vy) == pytest.approx(expected, rel=1e-9)
            assert vy == pytest.approx(0.0, abs=1e-9)

    def test_it_ignores_where_the_effector_actually_is(self):
        # This is the point of a feedforward launch: the command is the same
        # whatever else is pushing, so it is decisive and identical every trial.
        stuck = min_jerk.step(_plan(), 0.2, 0.0, 0.0, movement_allowed=True)
        dragged = min_jerk.step(_plan(), 0.2, -80.0, 55.0, movement_allowed=True)
        assert stuck == dragged

    def test_a_stalled_effector_still_gets_a_decisive_command(self):
        # The failure the old progress-keyed generator had: with the effector
        # held at the start, commanded speed collapsed to ~0 and nothing could
        # un-stall it. Keying the launch on time removes the deadlock outright.
        state = _plan(peak_speed=800.0)
        speeds = [math.hypot(*min_jerk.step(state, e, 0.0, 0.0, movement_allowed=True)) for e in (0.05, 0.1, 0.2, 0.3)]
        assert min(speeds) > 0.05 * 800.0

    def test_zero_duration_replans_from_the_first_step(self):
        state = _plan(ballistic_duration=0.0)
        min_jerk.step(state, 0.01, 0.0, 0.0, movement_allowed=True)
        # Re-planning consumed the step: the command is no longer the pure
        # feedforward bell value, which at t=0.01 is essentially zero.
        assert math.hypot(state.cmd_vx, state.cmd_vy) > 0.0

    def test_a_duration_past_the_reach_stays_feedforward_throughout(self):
        state = _plan(ballistic_duration=10.0)
        trace = _run(state, duration=NOMINAL_T)
        # Pure feedforward from rest reproduces canonical min-jerk under Euler.
        for t, x, _y, _vx, _vy in trace:
            assert x == pytest.approx(_canonical_position(t, NOMINAL_T, 0.0, DISTANCE), abs=5.0)


class TestUndisturbedReplanning:
    def test_replanning_reproduces_the_original_trajectory(self):
        # The load-bearing claim: the tail of a min-jerk trajectory is the
        # min-jerk solution for the tail, so re-planning from a point on your
        # own trajectory changes nothing. If this drifts, every "partial
        # compensation" property below is built on sand.
        state = _plan(ballistic_duration=0.3)
        trace = _run(state, dt=1.0 / 200.0, duration=NOMINAL_T)
        for t, x, y, _vx, _vy in trace:
            if t > NOMINAL_T - state.min_horizon:
                continue  # horizon floor takes over; see the test below
            assert x == pytest.approx(_canonical_position(t, NOMINAL_T, 0.0, DISTANCE), abs=1.5)
            assert y == pytest.approx(0.0, abs=1e-9)

    def test_replanning_tracks_as_well_as_pure_feedforward_does(self):
        # The sharper form of the claim above, self-calibrated against the
        # integrator: whatever error forward Euler contributes, re-planning must
        # add only a small fraction on top rather than accumulating its own.
        def worst_deviation(ballistic_duration):
            state = _plan(ballistic_duration=ballistic_duration)
            trace = _run(state, dt=1.0 / 200.0, duration=NOMINAL_T)
            return max(
                abs(x - _canonical_position(t, NOMINAL_T, 0.0, DISTANCE))
                for t, x, _y, _vx, _vy in trace
                if t <= NOMINAL_T - state.min_horizon
            )

        euler_floor = worst_deviation(10.0)  # never re-plans
        replanned = worst_deviation(0.3)
        assert replanned < euler_floor + 0.005 * DISTANCE

    def test_it_lands_on_the_target_and_stays(self):
        state = _plan(ballistic_duration=0.3)
        trace = _run(state, duration=4.0)
        final_x, final_y = trace[-1][1], trace[-1][2]
        assert final_x == pytest.approx(DISTANCE, abs=1.0)
        assert final_y == pytest.approx(0.0, abs=1e-6)
        # And has stopped asking for motion, rather than orbiting.
        assert math.hypot(trace[-1][3], trace[-1][4]) < 1.0

    def test_it_decelerates_into_the_target_rather_than_arriving_at_speed(self):
        state = _plan(ballistic_duration=0.3)
        trace = _run(state, duration=NOMINAL_T)
        speeds = [math.hypot(vx, vy) for _t, _x, _y, vx, vy in trace]
        peak = max(speeds)
        assert speeds[-1] < 0.2 * peak

    def test_peak_speed_is_close_to_the_requested_peak(self):
        state = _plan(ballistic_duration=0.3)
        trace = _run(state, dt=1.0 / 200.0, duration=NOMINAL_T)
        peak = max(math.hypot(vx, vy) for _t, _x, _y, vx, vy in trace)
        assert peak == pytest.approx(PEAK_SPEED, rel=0.05)

    def test_it_does_not_meaningfully_overshoot(self):
        state = _plan(ballistic_duration=0.3)
        trace = _run(state, duration=4.0)
        assert max(x for _t, x, _y, _vx, _vy in trace) < DISTANCE * 1.02


class TestDisturbedReplanning:
    def _perturbed_step(self, offset_y, *, elapsed=0.9):
        """One re-planned command after yanking the effector off-axis."""
        state = _plan(ballistic_duration=0.3)
        dt = 1.0 / 100.0
        x, y = 0.0, 0.0
        for i in range(int(elapsed / dt)):
            vx, vy = min_jerk.step(state, (i + 1) * dt, x, y, movement_allowed=True)
            x += vx * dt
            y += vy * dt
        y += offset_y
        vx, vy = min_jerk.step(state, elapsed + dt, x, y, movement_allowed=True)
        return state, x, y, vx, vy

    def test_it_compensates_partially_rather_than_snapping_to_the_target(self):
        # A fresh plan cannot contain a velocity discontinuity, so the command
        # turns toward the target without jumping to its bearing. That partial
        # correction is what a blend weight used to have to approximate.
        _state, x, y, vx, vy = self._perturbed_step(60.0)
        bearing = math.atan2(-y, DISTANCE - x)
        commanded = math.atan2(vy, vx)
        assert abs(commanded) > 1e-6, "must correct at all"
        assert abs(commanded) < abs(bearing), "must not snap to the target bearing"
        # Correcting the right way: pushed +y, the command must aim -y.
        assert vy < 0.0

    def test_a_bigger_disturbance_gets_a_bigger_correction(self):
        _s, _x, _y, _vx, small_vy = self._perturbed_step(20.0)
        _s, _x, _y, _vx, large_vy = self._perturbed_step(60.0)
        assert large_vy < small_vy < 0.0

    def test_less_time_left_means_a_harder_correction(self):
        # How hard it corrects is set by the remaining horizon, not by a tuning
        # constant -- the same disturbance late in a reach gets more urgency.
        _s, _x, _y, _vx, early_vy = self._perturbed_step(50.0, elapsed=0.5)
        _s, _x, _y, _vx, late_vy = self._perturbed_step(50.0, elapsed=1.5)
        assert late_vy < early_vy < 0.0

    def test_it_recovers_onto_the_target_after_a_disturbance(self):
        state = _plan(ballistic_duration=0.3)

        def yank(i, x, y):
            return (x, y + 80.0) if i == 90 else (x, y)

        trace = _run(state, duration=5.0, perturb=yank)
        final_x, final_y = trace[-1][1], trace[-1][2]
        assert math.hypot(final_x - DISTANCE, final_y) < 2.0

    def test_it_recovers_from_being_dragged_backwards(self):
        state = _plan(ballistic_duration=0.3)

        def drag(i, x, y):
            return (x - 150.0, y) if i == 100 else (x, y)

        trace = _run(state, duration=6.0, perturb=drag)
        final_x, final_y = trace[-1][1], trace[-1][2]
        assert math.hypot(final_x - DISTANCE, final_y) < 2.0

    def test_a_held_effector_keeps_commanding_motion(self):
        # Shared control's worst case: something pins the effector. The command
        # must not decay to nothing, or it can never win the argument.
        state = _plan(ballistic_duration=0.3)
        trace = _run(state, duration=3.0, perturb=lambda i, x, y: (0.0, 0.0))
        late = [math.hypot(vx, vy) for _t, _x, _y, vx, vy in trace[-50:]]
        assert min(late) > 0.0


class TestHorizonFloorAndClamp:
    def test_a_pinned_effector_does_not_demand_unbounded_speed(self):
        # Without a floored horizon this is the time-to-go singularity: distance
        # remaining, time running out, speed to infinity.
        state = _plan(ballistic_duration=0.3, max_speed_ratio=0.0)
        trace = _run(state, duration=8.0, perturb=lambda i, x, y: (0.0, 0.0))
        peak = max(math.hypot(vx, vy) for _t, _x, _y, vx, vy in trace)
        assert peak < 20.0 * PEAK_SPEED

    def test_the_speed_clamp_bounds_the_command(self):
        state = _plan(ballistic_duration=0.3, max_speed_ratio=1.5)
        trace = _run(state, duration=8.0, perturb=lambda i, x, y: (0.0, 0.0))
        peak = max(math.hypot(vx, vy) for _t, _x, _y, vx, vy in trace)
        assert peak <= 1.5 * PEAK_SPEED + 1e-6

    def test_the_clamp_is_off_when_the_ratio_is_zero(self):
        assert _plan(max_speed_ratio=0.0).max_speed == 0.0

    def test_the_horizon_floor_is_configurable(self):
        assert _plan(min_horizon=0.75).min_horizon == pytest.approx(0.75)


class TestReplanSeed:
    def test_commanded_is_the_default(self):
        assert _plan().seed is ReplanSeed.COMMANDED

    def test_commanded_ignores_a_disturbance_when_choosing_its_next_velocity(self):
        # Continuity is with the plan's own intent: the disturbance moved the
        # effector, and the position feedback reflects that, but it did not move
        # what the plan wants to be doing.
        state = _plan(ballistic_duration=0.0, seed=ReplanSeed.COMMANDED)
        min_jerk.step(state, 0.01, 0.0, 0.0, movement_allowed=True)
        before = (state.cmd_vx, state.cmd_vy)
        # Same position twice: a MEASURED seed would read zero velocity here.
        min_jerk.step(state, 0.02, 0.0, 0.0, movement_allowed=True)
        assert math.hypot(*before) > 0.0
        assert state.cmd_vx > 0.0

    def test_measured_reads_velocity_from_successive_positions(self):
        state = _plan(ballistic_duration=0.0, seed=ReplanSeed.MEASURED)
        min_jerk.step(state, 0.01, 0.0, 0.0, movement_allowed=True)
        # Effector visibly moving +y despite the plan asking for +x.
        vx_a, vy_a = min_jerk.step(state, 0.02, 0.0, 40.0, movement_allowed=True)
        state_b = _plan(ballistic_duration=0.0, seed=ReplanSeed.COMMANDED)
        min_jerk.step(state_b, 0.01, 0.0, 0.0, movement_allowed=True)
        vx_b, vy_b = min_jerk.step(state_b, 0.02, 0.0, 40.0, movement_allowed=True)
        # The measured seed inherits the observed +y motion; the commanded one
        # does not, so the two disagree about the vertical command.
        assert vy_a != pytest.approx(vy_b)
        assert vy_a > vy_b

    def test_both_seeds_agree_when_the_effector_follows_the_command(self):
        # No disturbance means measured and commanded velocity are the same
        # thing, so the modelling choice is invisible -- as it should be.
        traces = []
        for seed in (ReplanSeed.COMMANDED, ReplanSeed.MEASURED):
            state = _plan(ballistic_duration=0.3, seed=seed)
            traces.append(_run(state, dt=1.0 / 200.0, duration=NOMINAL_T))
        for (_t, xa, ya, _va, _vb), (_t2, xb, yb, _vc, _vd) in zip(*traces):
            assert xa == pytest.approx(xb, abs=1.0)
            assert ya == pytest.approx(yb, abs=1e-6)


class TestDeterminism:
    def test_the_same_epoch_and_timestamps_give_the_same_commands(self):
        # Two processes handed the same cue time and the same sample timestamps
        # must agree without exchanging any state.
        a, b = _plan(t0=1234.5), _plan(t0=1234.5)
        x = y = 0.0
        for i in range(200):
            t = 1234.5 + (i + 1) * 0.01
            va = min_jerk.step(a, t, x, y, movement_allowed=True)
            vb = min_jerk.step(b, t, x, y, movement_allowed=True)
            assert va == vb
            x += va[0] * 0.01
            y += va[1] * 0.01

    def test_a_repeated_timestamp_returns_the_previous_command(self):
        state = _plan(ballistic_duration=0.0)
        first = min_jerk.step(state, 0.01, 0.0, 0.0, movement_allowed=True)
        assert min_jerk.step(state, 0.01, 0.0, 0.0, movement_allowed=True) == first

    def test_the_timebase_offset_does_not_matter(self):
        zero, offset = _plan(t0=0.0), _plan(t0=1e6)
        x = y = 0.0
        xo = yo = 0.0
        for i in range(150):
            dt = 0.01
            vz = min_jerk.step(zero, (i + 1) * dt, x, y, movement_allowed=True)
            vo = min_jerk.step(offset, 1e6 + (i + 1) * dt, xo, yo, movement_allowed=True)
            assert vz == pytest.approx(vo)
            x += vz[0] * dt
            y += vz[1] * dt
            xo += vo[0] * dt
            yo += vo[1] * dt
