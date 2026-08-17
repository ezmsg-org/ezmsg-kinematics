"""Contract for the progress-keyed minimum-jerk reach generator.

The on-axis test is the important one: integrating the returned velocity must
recover the canonical min-jerk position curve ``10τ³ - 15τ⁴ + 6τ⁵``. Everything
else pins the properties that make the generator safe to run in more than one
place at once -- that it is a pure function of the measured position, that it
re-aims rather than fights a perturbation, and that it settles on the target
rather than orbiting it.
"""

from __future__ import annotations

import math

import pytest

from ezmsg.kinematics import min_jerk


def _canonical_min_jerk_position(t: float, T: float, start: float, end: float) -> float:
    """Closed-form min-jerk position at time ``t`` for reach ``start→end``."""
    if t <= 0.0:
        return start
    if t >= T:
        return end
    tau = t / T
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    return start + (end - start) * s


def _simulate(
    *,
    start_xy: tuple[float, float],
    target_xy: tuple[float, float],
    peak_speed: float,
    dt: float,
    n_steps: int,
) -> list[tuple[float, float]]:
    """Run the generator as the sole driver of the effector, forward-Euler."""
    state = min_jerk.MinJerkReach()
    min_jerk.begin_reach(
        state,
        start_xy[0],
        start_xy[1],
        target_xy[0],
        target_xy[1],
        peak_speed,
    )
    x, y = start_xy
    trace = [(x, y)]
    for _ in range(n_steps):
        vx, vy = min_jerk.step(state, x, y, movement_allowed=True)
        x += vx * dt
        y += vy * dt
        trace.append((x, y))
    return trace


class TestPlanning:
    def test_duration_is_set_so_the_bell_peaks_at_peak_speed(self):
        state = min_jerk.MinJerkReach()
        min_jerk.begin_reach(state, 0.0, 0.0, 300.0, 0.0, 300.0)
        # T = 1.875 * D / peak_speed, so peak = (D/T) * 1.875 = peak_speed.
        assert state.D == pytest.approx(300.0)
        assert state.T == pytest.approx(1.875)
        assert (state.D / state.T) * 1.875 == pytest.approx(300.0)
        assert state.active is True

    def test_a_longer_reach_takes_longer_rather_than_going_faster(self):
        near, far = min_jerk.MinJerkReach(), min_jerk.MinJerkReach()
        min_jerk.begin_reach(near, 0.0, 0.0, 100.0, 0.0, 300.0)
        min_jerk.begin_reach(far, 0.0, 0.0, 400.0, 0.0, 300.0)
        assert far.T == pytest.approx(4.0 * near.T)

    def test_a_zero_distance_reach_is_inactive_rather_than_an_error(self):
        state = min_jerk.MinJerkReach()
        min_jerk.begin_reach(state, 50.0, 50.0, 50.0, 50.0, 300.0)
        assert state.active is False
        assert min_jerk.step(state, 50.0, 50.0, movement_allowed=True) == (0.0, 0.0)

    def test_a_non_positive_peak_speed_is_inactive_rather_than_an_error(self):
        state = min_jerk.MinJerkReach()
        min_jerk.begin_reach(state, 0.0, 0.0, 300.0, 0.0, 0.0)
        assert state.active is False

    def test_beginning_a_new_reach_replans_from_where_the_effector_is(self):
        state = min_jerk.MinJerkReach()
        min_jerk.begin_reach(state, 0.0, 0.0, 300.0, 0.0, 300.0)
        x, y = 0.0, 0.0
        for _ in range(20):
            vx, vy = min_jerk.step(state, x, y, movement_allowed=True)
            x += vx / 60.0
            y += vy / 60.0

        min_jerk.begin_reach(state, x, y, x, y + 400.0, 300.0)
        assert state.start_x == pytest.approx(x)
        assert state.start_y == pytest.approx(y)
        assert state.D == pytest.approx(400.0)
        assert state.T == pytest.approx(1.875 * 400.0 / 300.0)
        # Must move immediately -- no cold-stick on the first step after replan.
        assert math.hypot(*min_jerk.step(state, x, y, movement_allowed=True)) > 0.0


class TestGating:
    def test_no_velocity_and_no_state_change_when_movement_disallowed(self):
        state = min_jerk.MinJerkReach()
        min_jerk.begin_reach(state, 0.0, 0.0, 300.0, 0.0, 300.0)
        tau_before = state.tau
        assert min_jerk.step(state, 0.0, 0.0, movement_allowed=False) == (0.0, 0.0)
        assert state.tau == tau_before

    def test_no_velocity_before_a_reach_is_planned(self):
        state = min_jerk.MinJerkReach()
        assert min_jerk.step(state, 0.0, 0.0, movement_allowed=True) == (0.0, 0.0)

    def test_reset_stops_the_reach(self):
        state = min_jerk.MinJerkReach()
        min_jerk.begin_reach(state, 0.0, 0.0, 300.0, 0.0, 300.0)
        min_jerk.reset(state)
        assert state.active is False
        assert min_jerk.step(state, 0.0, 0.0, movement_allowed=True) == (0.0, 0.0)

    def test_no_velocity_once_the_effector_is_on_the_target(self):
        state = min_jerk.MinJerkReach()
        min_jerk.begin_reach(state, 0.0, 0.0, 300.0, 0.0, 300.0)
        assert min_jerk.step(state, 300.0, 0.0, movement_allowed=True) == (0.0, 0.0)


class TestSpeedSchedule:
    def test_speed_at_a_given_progress_matches_the_closed_form_bell(self):
        # Placing the effector where canonical min-jerk would be at some τ must
        # recover that τ and its speed. This is the invariant any reimplementation
        # has to reproduce for two instances to agree.
        state = min_jerk.MinJerkReach()
        min_jerk.begin_reach(state, 0.0, 0.0, 300.0, 0.0, 300.0)
        for tau in (0.1, 0.25, 0.5, 0.75, 0.9):
            s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
            x = state.start_x + s * (state.target_x - state.start_x)
            y = state.start_y + s * (state.target_y - state.start_y)
            vx, vy = min_jerk.step(state, x, y, movement_allowed=True)
            expected = (state.D / state.T) * 30.0 * tau**2 * (1.0 - tau) ** 2
            assert math.hypot(vx, vy) == pytest.approx(expected, rel=1e-9)
            assert vy == pytest.approx(0.0, abs=1e-9)
            assert vx > 0.0

    def test_the_resolved_progress_does_not_depend_on_the_previous_step(self):
        # Bisection is unconditional, so a stale tau cannot bias the result --
        # which is what lets two instances at different rates agree.
        a, b = min_jerk.MinJerkReach(), min_jerk.MinJerkReach()
        min_jerk.begin_reach(a, 0.0, 0.0, 300.0, 0.0, 300.0)
        min_jerk.begin_reach(b, 0.0, 0.0, 300.0, 0.0, 300.0)
        a.tau = 0.0
        b.tau = 0.99
        assert min_jerk.step(a, 150.0, 0.0, movement_allowed=True) == min_jerk.step(
            b, 150.0, 0.0, movement_allowed=True
        )

    def test_peak_speed_over_a_full_reach_is_the_requested_peak(self):
        dt = 1.0 / 240.0
        peak_speed = 300.0
        state = min_jerk.MinJerkReach()
        min_jerk.begin_reach(state, 0.0, 0.0, 300.0, 0.0, peak_speed)
        x = y = 0.0
        peak = 0.0
        for _ in range(int(2.0 / dt)):
            vx, vy = min_jerk.step(state, x, y, movement_allowed=True)
            peak = max(peak, math.hypot(vx, vy))
            x += vx * dt
            y += vy * dt
        assert peak == pytest.approx(peak_speed, rel=0.01)

    def test_integrating_the_velocity_recovers_the_canonical_curve(self):
        dt = 1.0 / 60.0
        peak_speed = 300.0
        start_xy = (100.0, 200.0)
        target_xy = (400.0, 200.0)
        T = 1.875 * 300.0 / peak_speed
        # Progress-keyed inversion takes its cue from measured position alone,
        # with no time-based kick on the first step, so forward Euler at 60 Hz
        # lags the continuous curve by a few units around peak speed before
        # converging on the target. 5 is generous headroom.
        tolerance = 5.0

        trace = _simulate(
            start_xy=start_xy,
            target_xy=target_xy,
            peak_speed=peak_speed,
            dt=dt,
            n_steps=math.ceil(T / dt) + 2,
        )

        for i, (x, y) in enumerate(trace):
            t = i * dt
            assert x == pytest.approx(
                _canonical_min_jerk_position(t, T, start_xy[0], target_xy[0]),
                abs=tolerance,
            ), f"step {i} (t={t:.4f}s): x={x:.3f}"
            assert y == pytest.approx(start_xy[1], abs=1e-6)

        assert trace[-1][0] == pytest.approx(target_xy[0], abs=tolerance)

    def test_the_effector_settles_on_the_target_without_orbiting(self):
        dt = 1.0 / 60.0
        distance = 300.0
        trace = _simulate(
            start_xy=(0.0, 0.0),
            target_xy=(distance, 0.0),
            peak_speed=300.0,
            dt=dt,
            n_steps=int(4.0 / dt),  # well past T
        )
        # The schedule itself cannot overshoot -- progress past the target ends
        # the bell -- but a discrete integrator carries a fraction of the last
        # step past it. That excursion must stay negligible and must decay,
        # rather than the command turning around and driving back and forth.
        overshoot = max(x for x, _ in trace) - distance
        assert 0.0 <= overshoot < 0.001 * distance
        assert trace[-1][0] == pytest.approx(distance, abs=0.001 * distance)


class TestPerturbation:
    def test_the_command_re_aims_from_wherever_the_effector_now_is(self):
        dt = 1.0 / 60.0
        state = min_jerk.MinJerkReach()
        min_jerk.begin_reach(state, 0.0, 0.0, 300.0, 0.0, 300.0)
        x, y = 0.0, 0.0
        for _ in range(int((state.T / 2) / dt)):
            vx, vy = min_jerk.step(state, x, y, movement_allowed=True)
            x += vx * dt
            y += vy * dt

        y += 50.0  # yank it sideways
        vx, vy = min_jerk.step(state, x, y, movement_allowed=True)

        # Velocity must be parallel to (target - position), not to the original
        # axis: the generator re-aims instead of steering back to the old line.
        dx_aim, dy_aim = 300.0 - x, 0.0 - y
        cross = vx * dy_aim - vy * dx_aim
        norm = math.hypot(vx, vy) * math.hypot(dx_aim, dy_aim)
        assert abs(cross) / (norm + 1e-9) < 1e-6

    def test_the_output_depends_only_on_the_measured_position(self):
        # Two instances that saw completely different histories must agree once
        # they see the same position. This is what lets separate processes run
        # their own generator without exchanging state.
        travelled = min_jerk.MinJerkReach()
        fresh = min_jerk.MinJerkReach()
        for state in (travelled, fresh):
            min_jerk.begin_reach(state, 0.0, 0.0, 300.0, 0.0, 300.0)
        x = y = 0.0
        for _ in range(37):
            vx, vy = min_jerk.step(travelled, x, y, movement_allowed=True)
            x += vx / 60.0
            y += vy / 60.0
        assert min_jerk.step(travelled, 123.0, 4.0, movement_allowed=True) == min_jerk.step(
            fresh, 123.0, 4.0, movement_allowed=True
        )


class TestLaunchFloor:
    def test_it_is_off_unless_asked_for(self):
        state = min_jerk.MinJerkReach()
        assert state.launch_floor == 0.0 and state.ease_radius == 0.0
        min_jerk.begin_reach(state, 0.0, 0.0, 300.0, 0.0, 300.0)
        assert state.launch_floor == 0.0 and state.ease_radius == 0.0

    def test_zero_reproduces_the_plain_schedule_exactly(self):
        plain, floored = min_jerk.MinJerkReach(), min_jerk.MinJerkReach()
        min_jerk.begin_reach(plain, 0.0, 0.0, 300.0, 0.0, 800.0)
        min_jerk.begin_reach(floored, 0.0, 0.0, 300.0, 0.0, 800.0, launch_floor=0.0, ease_radius=0.0)
        for x in (5.0, 50.0, 150.0, 250.0, 299.0):
            assert min_jerk.step(plain, x, 0.0, movement_allowed=True) == min_jerk.step(
                floored, x, 0.0, movement_allowed=True
            )

    def test_it_breaks_the_stalled_effector_deadlock(self):
        # The failure it exists for: something else is driving, the effector has
        # not moved, so progress-keyed speed collapses to ~0 and the command can
        # never un-stall it.
        peak_speed, floor = 800.0, 150.0
        plain, floored = min_jerk.MinJerkReach(), min_jerk.MinJerkReach()
        min_jerk.begin_reach(plain, 0.0, 0.0, 300.0, 0.0, peak_speed)
        min_jerk.begin_reach(floored, 0.0, 0.0, 300.0, 0.0, peak_speed, launch_floor=floor, ease_radius=90.0)
        assert math.hypot(*min_jerk.step(plain, 0.0, 0.0, movement_allowed=True)) < 0.05 * peak_speed
        assert math.hypot(*min_jerk.step(floored, 0.0, 0.0, movement_allowed=True)) == pytest.approx(floor, rel=1e-6)

    def test_it_is_a_no_op_where_the_bell_is_already_higher(self):
        peak_speed, floor = 800.0, 150.0
        plain, floored = min_jerk.MinJerkReach(), min_jerk.MinJerkReach()
        min_jerk.begin_reach(plain, 0.0, 0.0, 300.0, 0.0, peak_speed)
        min_jerk.begin_reach(floored, 0.0, 0.0, 300.0, 0.0, peak_speed, launch_floor=floor, ease_radius=30.0)
        vp = math.hypot(*min_jerk.step(plain, 150.0, 0.0, movement_allowed=True))
        vf = math.hypot(*min_jerk.step(floored, 150.0, 0.0, movement_allowed=True))
        assert vp > floor
        assert vf == pytest.approx(vp, rel=1e-9)

    def test_it_releases_inside_the_ease_radius_so_the_reach_can_settle(self):
        state = min_jerk.MinJerkReach()
        min_jerk.begin_reach(state, 0.0, 0.0, 300.0, 0.0, 800.0, launch_floor=150.0, ease_radius=90.0)
        # Near the target the bell is far below the floor; holding the floor here
        # would orbit the target instead of settling on it.
        assert math.hypot(*min_jerk.step(state, 299.0, 0.0, movement_allowed=True)) < 150.0

    def test_the_floored_command_still_aims_at_the_target(self):
        state = min_jerk.MinJerkReach()
        min_jerk.begin_reach(state, 0.0, 0.0, 300.0, 0.0, 800.0, launch_floor=150.0, ease_radius=90.0)
        x, y = 50.0, 80.0  # off-axis and far from the target
        vx, vy = min_jerk.step(state, x, y, movement_allowed=True)
        dx_aim, dy_aim = 300.0 - x, 0.0 - y
        cross = vx * dy_aim - vy * dx_aim
        norm = math.hypot(vx, vy) * math.hypot(dx_aim, dy_aim)
        assert abs(cross) / (norm + 1e-9) < 1e-6
