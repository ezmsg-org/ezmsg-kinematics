"""Minimum-jerk reach generator, keyed on measured progress.

Stdlib-only, scalar in / scalar out, dataclass-backed state. Deliberately free
of any framework dependency so the same generator can run inside a streaming
graph, inside a task's render loop, or in an offline analysis, and produce the
same numbers in all three.

The classic minimum-jerk reach is a function of *time*: over a duration ``T``,
position follows ``p(t) = A + s(τ)·(B - A)`` with ``τ = t/T`` and
``s(τ) = 10τ³ - 15τ⁴ + 6τ⁵``, giving the familiar symmetric bell-shaped speed
profile. This module keys the same profile on *measured progress* instead:

* A reach is planned from a start point to a target, with a duration chosen so
  a caller-supplied ``peak_speed`` is the peak of the velocity bell.
* Each step, the effector's projection onto the start→target axis gives a
  progress fraction ``s ∈ [ε, 1-ε]``. Inverting ``s(τ)`` recovers the ``τ`` the
  effector has actually reached, whatever it did to get there.
* Speed comes from the velocity bell at that ``τ``; direction is the unit
  vector from the effector's *current* position toward the target.

Keying on progress rather than elapsed time makes the output a deterministic
function of a shared observable -- the measured position. Two processes
watching the same effector produce identical commands without exchanging any
state or agreeing on a clock. It also means the generator never fights a
perturbation: pushed off-axis, it re-aims from wherever it now is rather than
trying to return to a position it should have been at by now, and it cannot
overshoot, because progress past the target ends the bell.

The trade-off is the flip side of the same property: when the effector does not
move, progress does not advance, so the commanded speed stays near zero. That
is correct when this generator is what moves the effector. It deadlocks when
something else is (a closed loop in which the generator only advises), because
a stalled effector yields a ~0 command that cannot un-stall it. ``launch_floor``
exists for that case: it holds a minimum commanded speed toward the target while
farther than ``ease_radius`` from it, so there is always something to act on,
and releases near the target so the deceleration can still settle cleanly.
Both default to 0, which reproduces the pure progress-keyed schedule.

Typical use::

    reach = MinJerkReach()
    begin_reach(reach, *effector_xy, *target_xy, peak_speed=800.0)
    while reaching:
        vx, vy = step(reach, *effector_xy, movement_allowed=True)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["MinJerkReach", "begin_reach", "reset", "step"]


@dataclass
class MinJerkReach:
    """State for one minimum-jerk reach.

    Lifecycle:
        * :func:`begin_reach` starts (or restarts) a reach. Call it whenever
          movement becomes allowed and whenever the target moves mid-reach.
        * :func:`step` returns a velocity for the current effector position.
          While movement is gated off, pass ``movement_allowed=False`` -- the
          state is left frozen and the velocity is zero.
        * :func:`reset` is optional; it marks the reach inactive so a later
          :func:`step` returns zero even with ``movement_allowed=True``.

    Fields:
        start_x, start_y: effector position captured at ``begin_reach``.
        target_x, target_y: target captured at ``begin_reach``.
        D: straight-line distance from start to target.
        T: reach duration. Chosen so peak speed equals ``peak_speed``.
        tau: normalized min-jerk parameter, resolved from measured progress.
        launch_floor: minimum commanded speed while farther than
            ``ease_radius`` from the target. 0 disables.
        ease_radius: radius around the target within which ``launch_floor`` is
            released so the deceleration can settle.
        active: False before ``begin_reach`` or after ``reset``.
    """

    start_x: float = 0.0
    start_y: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    D: float = 0.0
    T: float = 0.0
    tau: float = 0.0
    launch_floor: float = 0.0
    ease_radius: float = 0.0
    active: bool = False


def begin_reach(
    state: MinJerkReach,
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    peak_speed: float,
    *,
    launch_floor: float = 0.0,
    ease_radius: float = 0.0,
) -> None:
    """(Re)initialize ``state`` for a reach from a start point to a target.

    ``peak_speed`` is the peak of the velocity bell, in whatever distance unit
    the caller works in, per second. The reach duration is derived from it and
    the distance, so the same ``peak_speed`` gives a longer reach proportionally
    longer to complete rather than a faster one.

    ``launch_floor`` / ``ease_radius`` configure the stall floor described in
    the module docstring; both default to 0 (disabled).

    A degenerate reach -- zero distance, or a non-positive ``peak_speed`` --
    leaves the state inactive rather than raising, so a caller driving this from
    live events does not have to pre-filter the case where a target coincides
    with the effector.
    """
    state.start_x = float(start_x)
    state.start_y = float(start_y)
    state.target_x = float(target_x)
    state.target_y = float(target_y)

    dx = state.target_x - state.start_x
    dy = state.target_y - state.start_y
    state.D = math.hypot(dx, dy)

    # The velocity bell peaks at 1.875 * D / T, so this choice of T makes
    # peak_speed the peak.
    if peak_speed > 0.0 and state.D > 0.0:
        state.T = 1.875 * state.D / float(peak_speed)
    else:
        state.T = 0.0

    # tau is recovered from measured progress on every step; nothing to seed.
    state.tau = 0.0
    state.launch_floor = float(launch_floor)
    state.ease_radius = float(ease_radius)

    state.active = state.D > 0.0 and state.T > 0.0


def reset(state: MinJerkReach) -> None:
    """Mark the reach inactive; subsequent :func:`step` calls return zero."""
    state.active = False


def step(
    state: MinJerkReach,
    pos_x: float,
    pos_y: float,
    movement_allowed: bool,
    *,
    eps: float = 1e-4,
    bisect_iters: int = 40,
) -> tuple[float, float]:
    """Velocity ``(vx, vy)`` for an effector currently at ``(pos_x, pos_y)``.

    Returns ``(0.0, 0.0)`` and leaves the state untouched when movement is not
    allowed, no reach is active, or the effector is exactly on the target.
    Otherwise resolves ``state.tau`` from measured progress and returns a vector
    aimed from the current position at the target, with min-jerk-shaped
    magnitude (floored by ``launch_floor`` while farther than ``ease_radius``).

    ``tau`` is resolved by **bisection** of ``s(τ) = s_measured`` on ``[0, 1]``,
    where ``s`` is monotonic. Bisection is unconditional and rate-agnostic:
    Newton's method on this quintic is unstable near ``τ=0``, where the cubic
    term makes the curve nearly flat, and can overshoot to the clamp when seeded
    from a previous step that is far from the solution. ``bisect_iters=40``
    resolves ``τ`` to ~``2⁻⁴⁰ ≈ 1e-12``, far below the integration error of any
    caller stepping this at a realistic rate.
    """
    if not movement_allowed or not state.active or state.T <= 0.0:
        return 0.0, 0.0

    # Progress along the reach: project (position - start) onto (target - start).
    ax = state.target_x - state.start_x
    ay = state.target_y - state.start_y
    axis_sq = ax * ax + ay * ay
    if axis_sq <= 0.0:
        return 0.0, 0.0

    s_raw = ((pos_x - state.start_x) * ax + (pos_y - state.start_y) * ay) / axis_sq
    # Clamped off both ends: at exactly 0 the bell is 0 and the reach could never
    # start; at exactly 1 it is 0 again and a slight overshoot would otherwise
    # read as negative progress.
    if s_raw < eps:
        s_measured = eps
    elif s_raw > 1.0 - eps:
        s_measured = 1.0 - eps
    else:
        s_measured = s_raw

    lo, hi = 0.0, 1.0
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        t3 = mid * mid * mid
        t4 = t3 * mid
        t5 = t4 * mid
        s_mid = 10.0 * t3 - 15.0 * t4 + 6.0 * t5
        if s_mid < s_measured:
            lo = mid
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    state.tau = tau

    one_minus = 1.0 - tau
    bell = 30.0 * tau * tau * one_minus * one_minus
    speed = (state.D / state.T) * bell

    # Direction is from the *current* position, not from the start: that is what
    # lets a perturbed effector re-aim instead of steering back to the original
    # line.
    dx = state.target_x - pos_x
    dy = state.target_y - pos_y
    dist = math.hypot(dx, dy)
    if dist <= 1e-9:
        return 0.0, 0.0

    if state.launch_floor > 0.0 and dist > state.ease_radius:
        speed = max(speed, state.launch_floor)

    return speed * (dx / dist), speed * (dy / dist)
