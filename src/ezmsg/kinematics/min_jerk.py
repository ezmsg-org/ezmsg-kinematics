"""Minimum-jerk reach generator with continuous re-planning.

Stdlib-only, scalar in / scalar out, dataclass-backed state. Deliberately free
of any framework dependency so the same generator can run inside a streaming
graph, inside a task's render loop, or in an offline analysis, and produce the
same numbers in all three.

The classic minimum-jerk reach is a function of time: over a duration ``T``,
position follows ``p(t) = A + s(τ)·(B - A)`` with ``τ = t/T`` and
``s(τ) = 10τ³ - 15τ⁴ + 6τ⁵``, giving the familiar symmetric bell-shaped speed
profile. That is the whole story when the generator is what moves the effector.
It is not, when something else is also pushing -- a decoder, a participant, a
disturbance -- because the effector then ends up somewhere the plan never
predicted, and a plan that only knows about elapsed time has nothing to say
about it.

This generator runs a reach in two phases.

**Ballistic.** For ``ballistic_duration`` after the reach starts, the command is
pure feedforward: the planned min-jerk velocity at the elapsed time, along the
original start→target axis, regardless of where the effector actually is. Real
reaches begin open-loop, and it means the launch command is decisive and
identical every time -- valuable if something downstream has to learn it.

**Re-planning.** After that, every step re-solves the minimum-jerk problem from
where the effector *is* to the target, and emits the beginning of that fresh
solution. The boundary conditions are the measured position, the current
commanded velocity and acceleration, and rest at the target.

Re-planning is what makes the response to a disturbance fall out rather than
have to be designed:

* **Undisturbed, it changes nothing.** The tail of a minimum-jerk trajectory is
  itself the minimum-jerk solution for the remaining problem, so re-planning
  from a point on your own trajectory reproduces it exactly. An effector that
  was never pushed follows the original reach, including its deceleration onto
  the target.
* **Disturbed, it compensates partially, and by the right amount.** A fresh plan
  cannot contain a velocity discontinuity, so it curves back toward the target
  rather than snapping to the new bearing. How hard it corrects is set by how
  much time is left, not by a blend weight anyone has to tune.
* **Speed and turning trade off on their own.** A large heading error forces the
  new polynomial to spend its early effort turning, which suppresses forward
  speed for as long as the turn takes.
* **It always eases in.** Rest at the target is a boundary condition of every
  re-plan, so there is no approach speed to tune and no overshoot to damp.

The remaining horizon is floored at ``min_horizon``. Without a floor, an
effector held away from the target as its deadline runs out demands unbounded
speed -- the time-to-go singularity. The floor costs exactness only in the final
``min_horizon`` of an undisturbed reach, where everything involved is already
near zero. ``max_speed_ratio`` clamps the command as a second line of defence.

Time is passed in rather than read from a clock, so a caller can drive this from
whatever timebase its data carries -- a stream's sample timestamps, a shared
network clock, or a recording being replayed offline. Two processes given the
same reach epoch and the same timestamps produce the same commands.

Typical use::

    reach = MinJerkReach()
    begin_reach(reach, cue_time, *effector_xy, *target_xy, peak_speed=800.0)
    while reaching:
        vx, vy = step(reach, now, *effector_xy, movement_allowed=True)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

__all__ = ["MinJerkReach", "ReplanSeed", "begin_reach", "reset", "step"]


class ReplanSeed(str, Enum):
    """Which velocity a re-plan treats as the effector's current velocity.

    This is a modelling choice, not a tuning knob, and it matters whenever
    something other than this generator is moving the effector.

    ``COMMANDED`` continues from the velocity this generator last asked for. The
    plan stays continuous with its own intent and treats the effector's position
    as feedback about where that intent ended up. Appropriate when the output
    represents what someone or something *wants* -- a disturbance moved the
    effector, but it did not move the intent. This is the default.

    ``MEASURED`` continues from the effector's observed velocity, differentiated
    from successive positions. The plan stays continuous with what is physically
    happening. Appropriate when the output drives the effector directly and you
    want the generator to own the real dynamics. Noisier, and if the output is
    being used as a training label it will carry whatever error the disturbance
    injected.
    """

    COMMANDED = "commanded"
    MEASURED = "measured"


@dataclass
class MinJerkReach:
    """State for one minimum-jerk reach.

    Lifecycle:
        * :func:`begin_reach` starts (or restarts) a reach. Call it whenever
          movement becomes allowed and whenever the target moves mid-reach.
        * :func:`step` returns a velocity for the current time and effector
          position. While movement is gated off, pass ``movement_allowed=False``
          -- the command goes to zero and the reach waits.
        * :func:`reset` is optional; it marks the reach inactive so a later
          :func:`step` returns zero even with ``movement_allowed=True``.

    Fields:
        t0: timestamp the reach started, in the caller's timebase.
        start_x, start_y: effector position captured at ``begin_reach``.
        target_x, target_y: target captured at ``begin_reach``.
        D: straight-line distance from start to target.
        T: nominal reach duration. Chosen so peak speed equals ``peak_speed``.
        ballistic_duration: how long the feedforward phase lasts.
        min_horizon: floor on the remaining time a re-plan is given.
        max_speed: absolute clamp on commanded speed. 0 disables.
        seed: which velocity a re-plan continues from.
        cmd_vx, cmd_vy: velocity last commanded -- the plan's own motor state.
        cmd_ax, cmd_ay: acceleration last commanded.
        last_x, last_y: effector position at the previous step, for the
            ``MEASURED`` seed.
        t_prev: timestamp of the previous step.
        active: False before ``begin_reach`` or after ``reset``.
    """

    t0: float = 0.0
    start_x: float = 0.0
    start_y: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    D: float = 0.0
    T: float = 0.0
    ballistic_duration: float = 0.5
    min_horizon: float = 0.3
    max_speed: float = 0.0
    seed: ReplanSeed = ReplanSeed.COMMANDED
    cmd_vx: float = 0.0
    cmd_vy: float = 0.0
    cmd_ax: float = 0.0
    cmd_ay: float = 0.0
    last_x: float = 0.0
    last_y: float = 0.0
    t_prev: float = 0.0
    active: bool = False


def begin_reach(
    state: MinJerkReach,
    t0: float,
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    peak_speed: float,
    *,
    ballistic_duration: float = 0.5,
    min_horizon: float = 0.3,
    max_speed_ratio: float = 2.0,
    seed: ReplanSeed = ReplanSeed.COMMANDED,
) -> None:
    """(Re)initialize ``state`` for a reach from a start point to a target.

    ``t0`` is when the reach begins, in whatever timebase the caller passes to
    :func:`step`. Anchoring to a timestamp the caller already has -- a cue
    event's own arrival time, say -- rather than to a clock read here is what
    lets two processes agree without exchanging state.

    ``peak_speed`` is the peak of the planned velocity bell, in the caller's
    distance unit per second. The duration is derived from it and the distance,
    so a longer reach takes proportionally longer rather than going faster.

    ``ballistic_duration`` is how long the command ignores the effector's
    position. 0 re-plans from the first step, which is well defined but gives up
    the identical, decisive launch. Values at or beyond the reach's own duration
    make it purely feedforward.

    ``min_horizon`` floors the time a re-plan is allowed to assume remains.
    ``max_speed_ratio`` clamps commanded speed to that multiple of
    ``peak_speed``; 0 disables the clamp. Both guard the same failure -- an
    effector held away from the target while its deadline expires.

    A degenerate reach -- zero distance, or a non-positive ``peak_speed`` --
    leaves the state inactive rather than raising, so a caller driving this from
    live events need not pre-filter a target that coincides with the effector.
    """
    state.t0 = float(t0)
    state.t_prev = float(t0)
    state.start_x = float(start_x)
    state.start_y = float(start_y)
    state.target_x = float(target_x)
    state.target_y = float(target_y)
    state.last_x = state.start_x
    state.last_y = state.start_y

    dx = state.target_x - state.start_x
    dy = state.target_y - state.start_y
    state.D = math.hypot(dx, dy)

    # The velocity bell peaks at 1.875 * D / T, so this choice of T makes
    # peak_speed the peak.
    if peak_speed > 0.0 and state.D > 0.0:
        state.T = 1.875 * state.D / float(peak_speed)
    else:
        state.T = 0.0

    state.ballistic_duration = max(0.0, float(ballistic_duration))
    state.min_horizon = max(1e-6, float(min_horizon))
    state.max_speed = max(0.0, float(max_speed_ratio)) * float(peak_speed)
    state.seed = seed

    # A reach starts from rest. Re-planning later continues from whatever the
    # ballistic phase handed over, but the plan's motor state begins at zero.
    state.cmd_vx = 0.0
    state.cmd_vy = 0.0
    state.cmd_ax = 0.0
    state.cmd_ay = 0.0

    state.active = state.D > 0.0 and state.T > 0.0


def reset(state: MinJerkReach) -> None:
    """Mark the reach inactive; subsequent :func:`step` calls return zero."""
    state.active = False


def _quintic_to_rest(p0: float, v0: float, a0: float, pf: float, T: float) -> tuple[float, float, float]:
    """Cubic..quintic coefficients taking ``(p0, v0, a0)`` to ``(pf, 0, 0)`` in ``T``.

    The minimum-jerk solution for fixed position/velocity/acceleration at both
    ends is the unique quintic satisfying them. With ``v0 = a0 = 0`` this
    reduces to the familiar ``10τ³ - 15τ⁴ + 6τ⁵``.
    """
    T2 = T * T
    T3 = T2 * T
    T4 = T2 * T2
    T5 = T4 * T

    # Residuals: what the already-determined terms fail to deliver at t = T.
    P = pf - p0 - v0 * T - 0.5 * a0 * T2
    V = -v0 - a0 * T
    A = -a0

    c3 = (10.0 * P - 4.0 * V * T + 0.5 * A * T2) / T3
    c4 = (-15.0 * P + 7.0 * V * T - A * T2) / T4
    c5 = (6.0 * P - 3.0 * V * T + 0.5 * A * T2) / T5
    return c3, c4, c5


def _eval_quintic(v0: float, a0: float, c3: float, c4: float, c5: float, t: float) -> tuple[float, float]:
    """Velocity and acceleration of the plan at time ``t`` from its start."""
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    v = v0 + a0 * t + 3.0 * c3 * t2 + 4.0 * c4 * t3 + 5.0 * c5 * t4
    a = a0 + 6.0 * c3 * t + 12.0 * c4 * t2 + 20.0 * c5 * t3
    return v, a


def _ballistic_command(state: MinJerkReach, elapsed: float) -> tuple[float, float, float, float]:
    """Planned velocity and acceleration at ``elapsed``, along the original axis."""
    tau = elapsed / state.T
    if tau < 0.0:
        tau = 0.0
    elif tau > 1.0:
        tau = 1.0

    # s'(τ) = 30τ²(1-τ)², s''(τ) = 60τ - 180τ² + 120τ³
    one_minus = 1.0 - tau
    ds = 30.0 * tau * tau * one_minus * one_minus
    dds = 60.0 * tau - 180.0 * tau * tau + 120.0 * tau * tau * tau

    ux = (state.target_x - state.start_x) / state.D
    uy = (state.target_y - state.start_y) / state.D
    speed = state.D * ds / state.T
    accel = state.D * dds / (state.T * state.T)
    return speed * ux, speed * uy, accel * ux, accel * uy


def step(
    state: MinJerkReach,
    t: float,
    pos_x: float,
    pos_y: float,
    movement_allowed: bool,
) -> tuple[float, float]:
    """Velocity ``(vx, vy)`` for an effector at ``(pos_x, pos_y)`` at time ``t``.

    ``t`` is in the same timebase as the ``t0`` given to :func:`begin_reach`.

    Returns ``(0.0, 0.0)`` when movement is not allowed or no reach is active.
    While movement is disallowed the command is held at rest and the clock
    reference is carried forward, so resuming does not produce a step
    proportional to how long the pause lasted.
    """
    if not movement_allowed or not state.active or state.T <= 0.0:
        if state.active:
            # Commanded rest, not merely "no output": whatever the effector does
            # while gated off, the plan is not asking for it.
            state.cmd_vx = state.cmd_vy = 0.0
            state.cmd_ax = state.cmd_ay = 0.0
            state.t_prev = float(t)
            state.last_x = float(pos_x)
            state.last_y = float(pos_y)
        return 0.0, 0.0

    elapsed = float(t) - state.t0
    if elapsed < 0.0:
        # A timestamp before the reach began; nothing has started yet.
        return 0.0, 0.0

    if elapsed < state.ballistic_duration:
        vx, vy, ax, ay = _ballistic_command(state, elapsed)
        state.cmd_vx, state.cmd_vy = vx, vy
        state.cmd_ax, state.cmd_ay = ax, ay
        state.t_prev = float(t)
        state.last_x, state.last_y = float(pos_x), float(pos_y)
        return vx, vy

    dt = float(t) - state.t_prev
    if dt <= 0.0:
        # Repeated or out-of-order timestamp: re-planning over a zero interval
        # is undefined, and the previous command is still the best answer.
        return state.cmd_vx, state.cmd_vy

    if state.seed is ReplanSeed.MEASURED:
        seed_vx = (float(pos_x) - state.last_x) / dt
        seed_vy = (float(pos_y) - state.last_y) / dt
    else:
        seed_vx, seed_vy = state.cmd_vx, state.cmd_vy

    # Floored so a stalled effector cannot demand unbounded speed as its
    # deadline expires.
    horizon = state.T - elapsed
    if horizon < state.min_horizon:
        horizon = state.min_horizon

    # Solved per axis: with rest at both ends this is the straight-line reach,
    # and with a nonzero starting velocity it curves, which is exactly the
    # partial compensation a disturbance should get.
    cx3, cx4, cx5 = _quintic_to_rest(float(pos_x), seed_vx, state.cmd_ax, state.target_x, horizon)
    cy3, cy4, cy5 = _quintic_to_rest(float(pos_y), seed_vy, state.cmd_ay, state.target_y, horizon)

    # Advance one control step along the fresh plan. Evaluating at 0 would just
    # return the seed velocity, which is why re-planning has to step forward.
    ahead = dt if dt < horizon else horizon
    vx, ax = _eval_quintic(seed_vx, state.cmd_ax, cx3, cx4, cx5, ahead)
    vy, ay = _eval_quintic(seed_vy, state.cmd_ay, cy3, cy4, cy5, ahead)

    if state.max_speed > 0.0:
        speed = math.hypot(vx, vy)
        if speed > state.max_speed:
            scale = state.max_speed / speed
            vx *= scale
            vy *= scale
            ax *= scale
            ay *= scale

    state.cmd_vx, state.cmd_vy = vx, vy
    state.cmd_ax, state.cmd_ay = ax, ay
    state.t_prev = float(t)
    state.last_x, state.last_y = float(pos_x), float(pos_y)
    return vx, vy
