# control/controllers.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Orientation controllers: proportional and PID with rate limiting."""

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

from .state import compute_orientation_error


class OrientationController(ABC):
    """Abstract base class for orientation controllers.

    All controllers share the ``update`` interface, which takes the current
    and target quaternions plus a time step and returns an angular velocity
    command vector.

    Args:
        max_angular_velocity: Maximum angular velocity magnitude
            (rad/s).  ``None`` means unlimited.
        max_angular_acceleration: Maximum angular acceleration magnitude
            (rad/s²).  ``None`` means unlimited.
    """

    def __init__(
        self,
        max_angular_velocity: Optional[float] = None,
        max_angular_acceleration: Optional[float] = None,
    ) -> None:
        self.max_angular_velocity = max_angular_velocity
        self.max_angular_acceleration = max_angular_acceleration
        self._prev_command: Optional[np.ndarray] = None

    @abstractmethod
    def compute_raw_command(
        self, current_q: ArrayLike, target_q: ArrayLike, dt: float
    ) -> np.ndarray:
        """Compute the raw (unclamped) angular velocity command.

        Subclasses must implement this.

        Args:
            current_q: Current orientation ``[x, y, z, w]``.
            target_q: Target orientation ``[x, y, z, w]``.
            dt: Time step in seconds.

        Returns:
            Raw angular velocity command ``(3,)`` in rad/s.
        """
        ...

    def update(
        self, current_q: ArrayLike, target_q: ArrayLike, dt: float
    ) -> np.ndarray:
        """Compute a rate-limited angular velocity command.

        Calls :meth:`compute_raw_command` and then applies
        :attr:`max_angular_velocity` and :attr:`max_angular_acceleration`
        limits.

        Args:
            current_q: Current orientation ``[x, y, z, w]``.
            target_q: Target orientation ``[x, y, z, w]``.
            dt: Time step in seconds.  Must be positive.

        Returns:
            Angular velocity command ``(3,)`` in rad/s.

        Raises:
            ValueError: If *dt* is not positive.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        command = self.compute_raw_command(current_q, target_q, dt)

        # Apply velocity limit
        if self.max_angular_velocity is not None:
            speed = np.linalg.norm(command)
            if speed > self.max_angular_velocity:
                command = command * (self.max_angular_velocity / speed)

        # Apply acceleration limit
        if self.max_angular_acceleration is not None and self._prev_command is not None:
            delta = command - self._prev_command
            accel = np.linalg.norm(delta) / dt
            if accel > self.max_angular_acceleration:
                max_delta = (
                    delta / np.linalg.norm(delta)
                    * self.max_angular_acceleration
                    * dt
                )
                command = self._prev_command + max_delta

        self._prev_command = command.copy()
        return command

    def reset(self) -> None:
        """Reset the controller's internal state."""
        self._prev_command = None


class ProportionalOrientationController(OrientationController):
    """Proportional (P) orientation controller.

    Computes the angular velocity command as ``kp × error``, where the
    error is the rotation vector from the current to the target orientation.

    Args:
        kp: Proportional gain.  Defaults to ``1.0``.
        max_angular_velocity: Maximum angular velocity magnitude (rad/s).
        max_angular_acceleration: Maximum angular acceleration (rad/s²).

    Example::

        from spinstep.control import ProportionalOrientationController

        ctrl = ProportionalOrientationController(kp=2.0, max_angular_velocity=3.14)
        cmd = ctrl.update([0, 0, 0, 1], [0, 0, 0.383, 0.924], dt=0.01)
    """

    def __init__(
        self,
        kp: float = 1.0,
        max_angular_velocity: Optional[float] = None,
        max_angular_acceleration: Optional[float] = None,
    ) -> None:
        super().__init__(max_angular_velocity, max_angular_acceleration)
        self.kp = kp

    def compute_raw_command(
        self, current_q: ArrayLike, target_q: ArrayLike, dt: float
    ) -> np.ndarray:
        """Compute ``kp × orientation_error``.

        Args:
            current_q: Current orientation ``[x, y, z, w]``.
            target_q: Target orientation ``[x, y, z, w]``.
            dt: Time step in seconds (unused by P controller).

        Returns:
            Angular velocity command ``(3,)`` in rad/s.
        """
        error = compute_orientation_error(current_q, target_q)
        return self.kp * error

    def reset(self) -> None:
        """Reset the controller's internal state."""
        super().reset()


class PIDOrientationController(OrientationController):
    """PID orientation controller with anti-windup.

    Computes the angular velocity command as
    ``kp × error + ki × ∫error·dt + kd × d(error)/dt``.
    Integral windup is prevented by clamping the integrated error magnitude
    to *max_integral*.

    Args:
        kp: Proportional gain.  Defaults to ``1.0``.
        ki: Integral gain.  Defaults to ``0.0``.
        kd: Derivative gain.  Defaults to ``0.0``.
        max_integral: Maximum magnitude of the integrated error vector.
            Defaults to ``10.0``.
        max_angular_velocity: Maximum angular velocity magnitude (rad/s).
        max_angular_acceleration: Maximum angular acceleration (rad/s²).

    Example::

        from spinstep.control import PIDOrientationController

        ctrl = PIDOrientationController(kp=2.0, ki=0.1, kd=0.5)
        cmd = ctrl.update([0, 0, 0, 1], [0, 0, 0.383, 0.924], dt=0.01)
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        max_integral: float = 10.0,
        max_angular_velocity: Optional[float] = None,
        max_angular_acceleration: Optional[float] = None,
    ) -> None:
        super().__init__(max_angular_velocity, max_angular_acceleration)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self._integral: np.ndarray = np.zeros(3)
        self._prev_error: Optional[np.ndarray] = None

    def compute_raw_command(
        self, current_q: ArrayLike, target_q: ArrayLike, dt: float
    ) -> np.ndarray:
        """Compute the PID angular velocity command.

        Args:
            current_q: Current orientation ``[x, y, z, w]``.
            target_q: Target orientation ``[x, y, z, w]``.
            dt: Time step in seconds.

        Returns:
            Angular velocity command ``(3,)`` in rad/s.
        """
        error = compute_orientation_error(current_q, target_q)

        # Proportional
        p_term = self.kp * error

        # Integral with anti-windup
        self._integral += error * dt
        integral_mag = np.linalg.norm(self._integral)
        if integral_mag > self.max_integral:
            self._integral = self._integral * (self.max_integral / integral_mag)
        i_term = self.ki * self._integral

        # Derivative
        if self._prev_error is not None:
            d_term = self.kd * (error - self._prev_error) / dt
        else:
            d_term = np.zeros(3)
        self._prev_error = error.copy()

        return p_term + i_term + d_term

    def reset(self) -> None:
        """Reset the controller's internal state (integral, derivative, etc.)."""
        super().reset()
        self._integral = np.zeros(3)
        self._prev_error = None
