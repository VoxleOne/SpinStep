# control/controllers.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Orientation controllers: proportional and PID with rate limiting.

All controllers operate in the observer-centered spherical model and
produce a :class:`~.state.ControlCommand` containing both angular
velocity and radial velocity components.
"""

from __future__ import annotations

__all__ = [
    "OrientationController",
    "ProportionalOrientationController",
    "PIDOrientationController",
]

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .state import ControlCommand, compute_orientation_error


class OrientationController(ABC):
    """Abstract base class for orientation controllers.

    All controllers share the :meth:`update` interface, which takes current
    and target poses (quaternion + distance) plus a time step, and returns
    a :class:`ControlCommand` with angular and radial velocity components.

    Args:
        max_angular_velocity: Maximum angular velocity magnitude
            (rad/s).  ``None`` means unlimited.
        max_angular_acceleration: Maximum angular acceleration magnitude
            (rad/s²).  ``None`` means unlimited.
        max_radial_velocity: Maximum radial speed (units/s).
            ``None`` means unlimited.
        max_radial_acceleration: Maximum radial acceleration (units/s²).
            ``None`` means unlimited.
    """

    def __init__(
        self,
        max_angular_velocity: Optional[float] = None,
        max_angular_acceleration: Optional[float] = None,
        max_radial_velocity: Optional[float] = None,
        max_radial_acceleration: Optional[float] = None,
    ) -> None:
        self.max_angular_velocity = max_angular_velocity
        self.max_angular_acceleration = max_angular_acceleration
        self.max_radial_velocity = max_radial_velocity
        self.max_radial_acceleration = max_radial_acceleration
        self._prev_angular_cmd: Optional[np.ndarray] = None
        self._prev_radial_cmd: Optional[float] = None

    @abstractmethod
    def compute_raw_command(
        self,
        current_q: ArrayLike,
        target_q: ArrayLike,
        dt: float,
        current_distance: float = 0.0,
        target_distance: float = 0.0,
    ) -> ControlCommand:
        """Compute the raw (unclamped) control command.

        Subclasses must implement this.

        Args:
            current_q: Current orientation ``[x, y, z, w]``.
            target_q: Target orientation ``[x, y, z, w]``.
            dt: Time step in seconds.
            current_distance: Current radial distance from observer.
            target_distance: Target radial distance from observer.

        Returns:
            Raw :class:`ControlCommand`.
        """
        ...

    def update(
        self,
        current_q: ArrayLike,
        target_q: ArrayLike,
        dt: float,
        current_distance: float = 0.0,
        target_distance: float = 0.0,
    ) -> ControlCommand:
        """Compute a rate-limited control command.

        Calls :meth:`compute_raw_command` and applies velocity and
        acceleration limits to both angular and radial components.

        Args:
            current_q: Current orientation ``[x, y, z, w]``.
            target_q: Target orientation ``[x, y, z, w]``.
            dt: Time step in seconds.  Must be positive.
            current_distance: Current radial distance from observer.
            target_distance: Target radial distance from observer.

        Returns:
            Rate-limited :class:`ControlCommand`.

        Raises:
            ValueError: If *dt* is not positive.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        raw = self.compute_raw_command(
            current_q, target_q, dt, current_distance, target_distance
        )
        angular = raw.angular_velocity.copy()
        radial = raw.radial_velocity

        # --- angular velocity limit ---
        if self.max_angular_velocity is not None:
            speed = np.linalg.norm(angular)
            if speed > self.max_angular_velocity:
                angular = angular * (self.max_angular_velocity / speed)

        # --- angular acceleration limit ---
        if (
            self.max_angular_acceleration is not None
            and self._prev_angular_cmd is not None
        ):
            delta = angular - self._prev_angular_cmd
            delta_norm = np.linalg.norm(delta)
            if delta_norm / dt > self.max_angular_acceleration:
                max_delta = (
                    delta / delta_norm * self.max_angular_acceleration * dt
                )
                angular = self._prev_angular_cmd + max_delta

        # --- radial velocity limit ---
        if self.max_radial_velocity is not None:
            if abs(radial) > self.max_radial_velocity:
                radial = np.sign(radial) * self.max_radial_velocity

        # --- radial acceleration limit ---
        if (
            self.max_radial_acceleration is not None
            and self._prev_radial_cmd is not None
        ):
            delta_r = radial - self._prev_radial_cmd
            if abs(delta_r) / dt > self.max_radial_acceleration:
                max_delta_r = np.sign(delta_r) * self.max_radial_acceleration * dt
                radial = self._prev_radial_cmd + max_delta_r

        self._prev_angular_cmd = angular.copy()
        self._prev_radial_cmd = float(radial)
        return ControlCommand(angular_velocity=angular, radial_velocity=radial)

    def reset(self) -> None:
        """Reset the controller's internal state."""
        self._prev_angular_cmd = None
        self._prev_radial_cmd = None


class ProportionalOrientationController(OrientationController):
    """Proportional (P) orientation controller.

    Computes angular velocity as ``kp × angular_error`` and radial
    velocity as ``kp_radial × radial_error``.

    Args:
        kp: Proportional gain for angular error.  Defaults to ``1.0``.
        kp_radial: Proportional gain for radial error.  Defaults to ``1.0``.
        max_angular_velocity: Maximum angular velocity (rad/s).
        max_angular_acceleration: Maximum angular acceleration (rad/s²).
        max_radial_velocity: Maximum radial speed (units/s).
        max_radial_acceleration: Maximum radial acceleration (units/s²).

    Example::

        from spinstep.control import ProportionalOrientationController

        ctrl = ProportionalOrientationController(kp=2.0, kp_radial=1.5)
        cmd = ctrl.update(
            [0, 0, 0, 1], [0, 0, 0.383, 0.924], dt=0.01,
            current_distance=3.0, target_distance=5.0,
        )
        print(cmd.angular_velocity, cmd.radial_velocity)
    """

    def __init__(
        self,
        kp: float = 1.0,
        kp_radial: float = 1.0,
        max_angular_velocity: Optional[float] = None,
        max_angular_acceleration: Optional[float] = None,
        max_radial_velocity: Optional[float] = None,
        max_radial_acceleration: Optional[float] = None,
    ) -> None:
        super().__init__(
            max_angular_velocity,
            max_angular_acceleration,
            max_radial_velocity,
            max_radial_acceleration,
        )
        self.kp = kp
        self.kp_radial = kp_radial

    def compute_raw_command(
        self,
        current_q: ArrayLike,
        target_q: ArrayLike,
        dt: float,
        current_distance: float = 0.0,
        target_distance: float = 0.0,
    ) -> ControlCommand:
        """Compute ``kp × error`` for angular and radial components.

        Args:
            current_q: Current orientation ``[x, y, z, w]``.
            target_q: Target orientation ``[x, y, z, w]``.
            dt: Time step in seconds (unused by P controller).
            current_distance: Current radial distance.
            target_distance: Target radial distance.

        Returns:
            :class:`ControlCommand`.
        """
        ang_error, rad_error = compute_orientation_error(
            current_q, target_q, current_distance, target_distance
        )
        return ControlCommand(
            angular_velocity=self.kp * ang_error,
            radial_velocity=self.kp_radial * rad_error,
        )

    def reset(self) -> None:
        """Reset the controller's internal state."""
        super().reset()


class PIDOrientationController(OrientationController):
    """PID orientation controller with anti-windup.

    Computes ``kp × e + ki × ∫e·dt + kd × de/dt`` for both the angular
    and radial error channels independently.  Integral windup is prevented
    by clamping integrated error magnitudes.

    Args:
        kp: Proportional gain (angular).
        ki: Integral gain (angular).
        kd: Derivative gain (angular).
        kp_radial: Proportional gain (radial).
        ki_radial: Integral gain (radial).
        kd_radial: Derivative gain (radial).
        max_integral: Maximum angular integral magnitude.
        max_integral_radial: Maximum radial integral magnitude.
        max_angular_velocity: Maximum angular velocity (rad/s).
        max_angular_acceleration: Maximum angular acceleration (rad/s²).
        max_radial_velocity: Maximum radial speed (units/s).
        max_radial_acceleration: Maximum radial acceleration (units/s²).

    Example::

        from spinstep.control import PIDOrientationController

        ctrl = PIDOrientationController(kp=2.0, ki=0.1, kd=0.5,
                                        kp_radial=1.0, ki_radial=0.05)
        cmd = ctrl.update(
            [0, 0, 0, 1], [0, 0, 0.383, 0.924], dt=0.01,
            current_distance=3.0, target_distance=5.0,
        )
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        kp_radial: float = 1.0,
        ki_radial: float = 0.0,
        kd_radial: float = 0.0,
        max_integral: float = 10.0,
        max_integral_radial: float = 10.0,
        max_angular_velocity: Optional[float] = None,
        max_angular_acceleration: Optional[float] = None,
        max_radial_velocity: Optional[float] = None,
        max_radial_acceleration: Optional[float] = None,
    ) -> None:
        super().__init__(
            max_angular_velocity,
            max_angular_acceleration,
            max_radial_velocity,
            max_radial_acceleration,
        )
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kp_radial = kp_radial
        self.ki_radial = ki_radial
        self.kd_radial = kd_radial
        self.max_integral = max_integral
        self.max_integral_radial = max_integral_radial

        self._ang_integral: np.ndarray = np.zeros(3)
        self._rad_integral: float = 0.0
        self._prev_ang_error: Optional[np.ndarray] = None
        self._prev_rad_error: Optional[float] = None

    def compute_raw_command(
        self,
        current_q: ArrayLike,
        target_q: ArrayLike,
        dt: float,
        current_distance: float = 0.0,
        target_distance: float = 0.0,
    ) -> ControlCommand:
        """Compute the PID command for angular and radial channels.

        Args:
            current_q: Current orientation ``[x, y, z, w]``.
            target_q: Target orientation ``[x, y, z, w]``.
            dt: Time step in seconds.
            current_distance: Current radial distance.
            target_distance: Target radial distance.

        Returns:
            :class:`ControlCommand`.
        """
        ang_error, rad_error = compute_orientation_error(
            current_q, target_q, current_distance, target_distance
        )

        # --- angular PID ---
        p_ang = self.kp * ang_error

        self._ang_integral += ang_error * dt
        mag = np.linalg.norm(self._ang_integral)
        if mag > self.max_integral:
            self._ang_integral *= self.max_integral / mag
        i_ang = self.ki * self._ang_integral

        if self._prev_ang_error is not None:
            d_ang = self.kd * (ang_error - self._prev_ang_error) / dt
        else:
            d_ang = np.zeros(3)
        self._prev_ang_error = ang_error.copy()

        angular_cmd = p_ang + i_ang + d_ang

        # --- radial PID ---
        p_rad = self.kp_radial * rad_error

        self._rad_integral += rad_error * dt
        if abs(self._rad_integral) > self.max_integral_radial:
            self._rad_integral = np.sign(self._rad_integral) * self.max_integral_radial
        i_rad = self.ki_radial * self._rad_integral

        if self._prev_rad_error is not None:
            d_rad = self.kd_radial * (rad_error - self._prev_rad_error) / dt
        else:
            d_rad = 0.0
        self._prev_rad_error = rad_error

        radial_cmd = p_rad + i_rad + d_rad

        return ControlCommand(
            angular_velocity=angular_cmd,
            radial_velocity=radial_cmd,
        )

    def reset(self) -> None:
        """Reset the controller's internal state (integral, derivative, etc.)."""
        super().reset()
        self._ang_integral = np.zeros(3)
        self._rad_integral = 0.0
        self._prev_ang_error = None
        self._prev_rad_error = None
