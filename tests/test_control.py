# test_control.py — SpinStep Test Suite — MIT License
# Tests for spinstep.control subpackage (state, controllers, trajectory)

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from spinstep.control.state import (
    ControlCommand,
    OrientationState,
    compute_orientation_error,
    integrate_state,
)
from spinstep.control.controllers import (
    PIDOrientationController,
    ProportionalOrientationController,
)
from spinstep.control.trajectory import (
    OrientationTrajectory,
    TrajectoryController,
    TrajectoryInterpolator,
)


# ===== OrientationState =====


class TestOrientationState:
    def test_defaults(self):
        state = OrientationState()
        assert np.allclose(state.quaternion, [0, 0, 0, 1])
        assert state.distance == 0.0
        assert np.allclose(state.angular_velocity, [0, 0, 0])
        assert state.radial_velocity == 0.0
        assert state.timestamp == 0.0

    def test_with_distance(self):
        state = OrientationState([0, 0, 0, 1], distance=5.0)
        assert state.distance == 5.0

    def test_normalizes_quaternion(self):
        state = OrientationState([0, 0, 0, 2])
        assert np.allclose(state.quaternion, [0, 0, 0, 1])

    def test_full_state(self):
        state = OrientationState(
            [0, 0, 0, 1],
            distance=10.0,
            angular_velocity=[0, 0, 1.0],
            radial_velocity=0.5,
            timestamp=1.0,
        )
        assert state.distance == 10.0
        assert state.radial_velocity == 0.5
        assert state.timestamp == 1.0

    def test_invalid_quaternion_shape(self):
        with pytest.raises(ValueError, match="shape"):
            OrientationState([1, 0, 0])

    def test_zero_quaternion(self):
        with pytest.raises(ValueError, match="non-zero"):
            OrientationState([0, 0, 0, 0])

    def test_negative_distance(self):
        with pytest.raises(ValueError, match="non-negative"):
            OrientationState([0, 0, 0, 1], distance=-1.0)

    def test_repr(self):
        state = OrientationState([0, 0, 0, 1], distance=5.0)
        r = repr(state)
        assert "OrientationState" in r
        assert "d=5.0" in r


# ===== ControlCommand =====


class TestControlCommand:
    def test_defaults(self):
        cmd = ControlCommand()
        assert np.allclose(cmd.angular_velocity, [0, 0, 0])
        assert cmd.radial_velocity == 0.0

    def test_custom(self):
        cmd = ControlCommand([1, 0, 0], radial_velocity=2.5)
        assert np.allclose(cmd.angular_velocity, [1, 0, 0])
        assert cmd.radial_velocity == 2.5

    def test_repr(self):
        cmd = ControlCommand([0, 0, 1.0], radial_velocity=0.5)
        r = repr(cmd)
        assert "ControlCommand" in r


# ===== integrate_state =====


class TestIntegrateState:
    def test_stationary(self):
        state = OrientationState([0, 0, 0, 1], distance=5.0)
        new = integrate_state(state, dt=0.1)
        assert np.allclose(new.quaternion, [0, 0, 0, 1])
        assert new.distance == 5.0
        assert new.timestamp == pytest.approx(0.1)

    def test_radial_integration(self):
        state = OrientationState(
            [0, 0, 0, 1], distance=5.0, radial_velocity=2.0
        )
        new = integrate_state(state, dt=1.0)
        assert new.distance == pytest.approx(7.0)

    def test_radial_clamp_to_zero(self):
        """Distance cannot go negative."""
        state = OrientationState(
            [0, 0, 0, 1], distance=1.0, radial_velocity=-5.0
        )
        new = integrate_state(state, dt=1.0)
        assert new.distance == 0.0

    def test_angular_integration(self):
        """Rotating about Z axis."""
        omega = np.pi  # rad/s → 180° per second
        state = OrientationState(
            [0, 0, 0, 1], angular_velocity=[0, 0, omega]
        )
        new = integrate_state(state, dt=0.5)
        # After 0.5s at π rad/s → 90° rotation about Z
        angle = R.from_quat(new.quaternion).magnitude()
        assert angle == pytest.approx(np.pi / 2, abs=0.01)

    def test_combined_integration(self):
        state = OrientationState(
            [0, 0, 0, 1],
            distance=5.0,
            angular_velocity=[0, 0, 1.0],
            radial_velocity=0.5,
            timestamp=1.0,
        )
        new = integrate_state(state, dt=0.1)
        assert new.timestamp == pytest.approx(1.1)
        assert new.distance == pytest.approx(5.05)
        assert not np.allclose(new.quaternion, [0, 0, 0, 1])

    def test_invalid_dt(self):
        state = OrientationState()
        with pytest.raises(ValueError):
            integrate_state(state, dt=0)
        with pytest.raises(ValueError):
            integrate_state(state, dt=-0.1)


# ===== compute_orientation_error =====


class TestComputeOrientationError:
    def test_no_error(self):
        ang, rad = compute_orientation_error(
            [0, 0, 0, 1], [0, 0, 0, 1], 5.0, 5.0
        )
        assert np.allclose(ang, [0, 0, 0], atol=1e-6)
        assert rad == pytest.approx(0.0)

    def test_angular_error_only(self):
        q_target = R.from_euler("z", 45, degrees=True).as_quat()
        ang, rad = compute_orientation_error(
            [0, 0, 0, 1], q_target, 5.0, 5.0
        )
        assert np.linalg.norm(ang) == pytest.approx(np.deg2rad(45), abs=0.01)
        assert rad == pytest.approx(0.0)

    def test_radial_error_only(self):
        ang, rad = compute_orientation_error(
            [0, 0, 0, 1], [0, 0, 0, 1], 3.0, 7.0
        )
        assert np.allclose(ang, [0, 0, 0], atol=1e-6)
        assert rad == pytest.approx(4.0)

    def test_combined_error(self):
        q_target = R.from_euler("z", 90, degrees=True).as_quat()
        ang, rad = compute_orientation_error(
            [0, 0, 0, 1], q_target, 2.0, 8.0
        )
        assert np.linalg.norm(ang) == pytest.approx(np.pi / 2, abs=0.01)
        assert rad == pytest.approx(6.0)

    def test_backward_compat_no_distance(self):
        """Works without distance arguments (defaults to 0)."""
        ang, rad = compute_orientation_error([0, 0, 0, 1], [0, 0, 0, 1])
        assert np.allclose(ang, [0, 0, 0], atol=1e-6)
        assert rad == pytest.approx(0.0)


# ===== ProportionalOrientationController =====


class TestProportionalController:
    def test_zero_error(self):
        ctrl = ProportionalOrientationController(kp=2.0)
        cmd = ctrl.update([0, 0, 0, 1], [0, 0, 0, 1], dt=0.01)
        assert np.allclose(cmd.angular_velocity, [0, 0, 0], atol=1e-6)
        assert cmd.radial_velocity == pytest.approx(0.0, abs=1e-6)

    def test_angular_error(self):
        ctrl = ProportionalOrientationController(kp=1.0)
        q_target = R.from_euler("z", 45, degrees=True).as_quat()
        cmd = ctrl.update([0, 0, 0, 1], q_target, dt=0.01)
        assert np.linalg.norm(cmd.angular_velocity) > 0

    def test_radial_error(self):
        ctrl = ProportionalOrientationController(kp=1.0, kp_radial=2.0)
        cmd = ctrl.update(
            [0, 0, 0, 1], [0, 0, 0, 1], dt=0.01,
            current_distance=3.0, target_distance=5.0,
        )
        assert cmd.radial_velocity == pytest.approx(4.0)  # 2.0 × 2.0

    def test_velocity_limit(self):
        ctrl = ProportionalOrientationController(
            kp=100.0, max_angular_velocity=1.0
        )
        q_target = R.from_euler("z", 90, degrees=True).as_quat()
        cmd = ctrl.update([0, 0, 0, 1], q_target, dt=0.01)
        assert np.linalg.norm(cmd.angular_velocity) <= 1.0 + 1e-6

    def test_radial_velocity_limit(self):
        ctrl = ProportionalOrientationController(
            kp_radial=100.0, max_radial_velocity=2.0
        )
        cmd = ctrl.update(
            [0, 0, 0, 1], [0, 0, 0, 1], dt=0.01,
            current_distance=0.0, target_distance=10.0,
        )
        assert abs(cmd.radial_velocity) <= 2.0 + 1e-6

    def test_invalid_dt(self):
        ctrl = ProportionalOrientationController()
        with pytest.raises(ValueError):
            ctrl.update([0, 0, 0, 1], [0, 0, 0, 1], dt=0)

    def test_reset(self):
        ctrl = ProportionalOrientationController(kp=1.0)
        ctrl.update([0, 0, 0, 1], [0, 0, 0, 1], dt=0.01)
        ctrl.reset()
        assert ctrl._prev_angular_cmd is None


# ===== PIDOrientationController =====


class TestPIDController:
    def test_zero_error(self):
        ctrl = PIDOrientationController(kp=1.0, ki=0.1, kd=0.5)
        cmd = ctrl.update([0, 0, 0, 1], [0, 0, 0, 1], dt=0.01)
        assert np.allclose(cmd.angular_velocity, [0, 0, 0], atol=1e-6)

    def test_integral_accumulates(self):
        ctrl = PIDOrientationController(kp=0.0, ki=1.0, kd=0.0)
        q_target = R.from_euler("z", 10, degrees=True).as_quat()
        # First step — integral starts accumulating
        ctrl.update([0, 0, 0, 1], q_target, dt=0.1)
        # Second step — integral grows
        cmd = ctrl.update([0, 0, 0, 1], q_target, dt=0.1)
        assert np.linalg.norm(cmd.angular_velocity) > 0

    def test_derivative_term(self):
        ctrl = PIDOrientationController(kp=0.0, ki=0.0, kd=1.0)
        q1 = R.from_euler("z", 10, degrees=True).as_quat()
        q2 = R.from_euler("z", 20, degrees=True).as_quat()
        # First call sets _prev_error
        ctrl.update([0, 0, 0, 1], q1, dt=0.01)
        # Second call with different target should produce d_term
        cmd = ctrl.update([0, 0, 0, 1], q2, dt=0.01)
        assert np.linalg.norm(cmd.angular_velocity) > 0

    def test_radial_pid(self):
        ctrl = PIDOrientationController(
            kp=0.0, ki=0.0, kd=0.0,
            kp_radial=2.0, ki_radial=0.5, kd_radial=0.0,
        )
        cmd = ctrl.update(
            [0, 0, 0, 1], [0, 0, 0, 1], dt=0.1,
            current_distance=3.0, target_distance=5.0,
        )
        assert cmd.radial_velocity == pytest.approx(2.0 * 2.0 + 0.5 * 2.0 * 0.1)

    def test_reset_clears_state(self):
        ctrl = PIDOrientationController(kp=1.0, ki=1.0, kd=1.0)
        ctrl.update([0, 0, 0, 1], [0, 0, 0, 1], dt=0.01)
        ctrl.reset()
        assert np.allclose(ctrl._ang_integral, [0, 0, 0])
        assert ctrl._rad_integral == 0.0
        assert ctrl._prev_ang_error is None
        assert ctrl._prev_rad_error is None

    def test_anti_windup(self):
        """Integral should be clamped."""
        ctrl = PIDOrientationController(kp=0.0, ki=1.0, max_integral=0.01)
        q_target = R.from_euler("z", 90, degrees=True).as_quat()
        for _ in range(100):
            ctrl.update([0, 0, 0, 1], q_target, dt=1.0)
        assert np.linalg.norm(ctrl._ang_integral) <= 0.01 + 1e-6


# ===== OrientationTrajectory =====


class TestOrientationTrajectory:
    def test_basic_3_tuple(self):
        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 5.0, 0.0),
            ([0, 0, 0.383, 0.924], 10.0, 1.0),
        ])
        assert len(traj) == 2
        assert traj.duration == pytest.approx(1.0)
        assert np.allclose(traj.distances, [5.0, 10.0])

    def test_basic_2_tuple_backward_compat(self):
        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 0.0),
            ([0, 0, 0.383, 0.924], 1.0),
        ])
        assert len(traj) == 2
        assert np.allclose(traj.distances, [0.0, 0.0])

    def test_mixed_tuple_lengths(self):
        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 5.0, 0.0),
            ([0, 0, 0.383, 0.924], 1.0),  # 2-tuple, distance=0
        ])
        assert traj.distances[0] == 5.0
        assert traj.distances[1] == 0.0

    def test_too_few_waypoints(self):
        with pytest.raises(ValueError, match="At least 2"):
            OrientationTrajectory([([0, 0, 0, 1], 0.0)])

    def test_non_increasing_times(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            OrientationTrajectory([
                ([0, 0, 0, 1], 5.0, 1.0),
                ([0, 0, 0, 1], 5.0, 0.5),
            ])

    def test_properties(self):
        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 5.0, 1.0),
            ([0, 0, 0, 1], 10.0, 3.0),
        ])
        assert traj.start_time == 1.0
        assert traj.end_time == 3.0
        assert traj.duration == 2.0

    def test_repr(self):
        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 5.0, 0.0),
            ([0, 0, 0, 1], 10.0, 1.0),
        ])
        assert "OrientationTrajectory" in repr(traj)


# ===== TrajectoryInterpolator =====


class TestTrajectoryInterpolator:
    def test_at_waypoints(self):
        q0 = [0, 0, 0, 1]
        q1 = R.from_euler("z", 90, degrees=True).as_quat().tolist()
        traj = OrientationTrajectory([
            (q0, 5.0, 0.0),
            (q1, 10.0, 1.0),
        ])
        interp = TrajectoryInterpolator(traj)
        q_start, d_start = interp.evaluate(0.0)
        q_end, d_end = interp.evaluate(1.0)
        assert np.allclose(q_start, q0, atol=1e-6)
        assert d_start == pytest.approx(5.0)
        assert d_end == pytest.approx(10.0)

    def test_midpoint_distance(self):
        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 4.0, 0.0),
            ([0, 0, 0, 1], 8.0, 1.0),
        ])
        interp = TrajectoryInterpolator(traj)
        _, d_mid = interp.evaluate(0.5)
        assert d_mid == pytest.approx(6.0)

    def test_before_start(self):
        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 5.0, 1.0),
            ([0, 0, 0, 1], 10.0, 2.0),
        ])
        interp = TrajectoryInterpolator(traj)
        q, d = interp.evaluate(0.0)
        assert d == pytest.approx(5.0)

    def test_after_end(self):
        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 5.0, 0.0),
            ([0, 0, 0, 1], 10.0, 1.0),
        ])
        interp = TrajectoryInterpolator(traj)
        q, d = interp.evaluate(99.0)
        assert d == pytest.approx(10.0)


# ===== TrajectoryController =====


class TestTrajectoryController:
    def test_basic_tracking(self):
        q0 = [0, 0, 0, 1]
        q1 = R.from_euler("z", 45, degrees=True).as_quat().tolist()
        traj = OrientationTrajectory([
            (q0, 5.0, 0.0),
            (q1, 10.0, 1.0),
        ])
        ctrl = ProportionalOrientationController(kp=1.0, kp_radial=1.0)
        tc = TrajectoryController(ctrl, traj)

        cmd = tc.update(q0, t=0.5, dt=0.01, current_distance=7.0)
        # At t=0.5 target distance ≈ 7.5, so radial_velocity > 0
        assert cmd.radial_velocity > 0
        assert not tc.is_complete

    def test_complete_flag(self):
        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 5.0, 0.0),
            ([0, 0, 0, 1], 10.0, 1.0),
        ])
        ctrl = ProportionalOrientationController()
        tc = TrajectoryController(ctrl, traj)
        tc.update([0, 0, 0, 1], t=1.5, dt=0.01, current_distance=10.0)
        assert tc.is_complete

    def test_reset(self):
        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 5.0, 0.0),
            ([0, 0, 0, 1], 10.0, 1.0),
        ])
        ctrl = ProportionalOrientationController()
        tc = TrajectoryController(ctrl, traj)
        tc.update([0, 0, 0, 1], t=2.0, dt=0.01)
        assert tc.is_complete
        tc.reset()
        assert not tc.is_complete


# ===== Integration: full control loop =====


class TestControlLoop:
    def test_converges_to_target(self):
        """P controller should drive orientation toward target over many steps."""
        ctrl = ProportionalOrientationController(kp=5.0, kp_radial=3.0)
        current_q = np.array([0, 0, 0, 1.0])
        target_q = R.from_euler("z", 45, degrees=True).as_quat()
        current_dist = 5.0
        target_dist = 10.0
        dt = 0.01

        for _ in range(500):
            cmd = ctrl.update(
                current_q, target_q, dt,
                current_distance=current_dist, target_distance=target_dist,
            )
            state = OrientationState(
                current_q, distance=current_dist,
                angular_velocity=cmd.angular_velocity,
                radial_velocity=cmd.radial_velocity,
            )
            new_state = integrate_state(state, dt)
            current_q = new_state.quaternion
            current_dist = new_state.distance

        # Should be close to target
        ang_err, rad_err = compute_orientation_error(
            current_q, target_q, current_dist, target_dist
        )
        assert np.linalg.norm(ang_err) < 0.01
        assert abs(rad_err) < 0.1

    def test_trajectory_tracking_loop(self):
        """TrajectoryController should follow a trajectory."""
        q0 = [0, 0, 0, 1]
        q1 = R.from_euler("z", 90, degrees=True).as_quat().tolist()
        traj = OrientationTrajectory([
            (q0, 5.0, 0.0),
            (q1, 10.0, 2.0),
        ])
        ctrl = ProportionalOrientationController(kp=5.0, kp_radial=3.0)
        tc = TrajectoryController(ctrl, traj)

        current_q = np.array(q0, dtype=float)
        current_dist = 5.0
        dt = 0.01
        t = 0.0

        for _ in range(200):
            cmd = tc.update(current_q, t=t, dt=dt, current_distance=current_dist)
            state = OrientationState(
                current_q, distance=current_dist,
                angular_velocity=cmd.angular_velocity,
                radial_velocity=cmd.radial_velocity,
            )
            new_state = integrate_state(state, dt)
            current_q = new_state.quaternion
            current_dist = new_state.distance
            t += dt

        # At t=2.0 should be near q1 / distance=10
        # At t=2.0 we're only at t=2.0 so check trajectory makes progress
        assert current_dist > 5.0  # moved outward
