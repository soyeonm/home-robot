import abc
from typing import Tuple

import numpy as np

V_MAX_DEFAULT = 0.2  # base.params["motion"]["default"]["vel_m"]
W_MAX_DEFAULT = 0.45  # (vel_m_max - vel_m_default) / wheel_separation_m
ACC_LIN = 0.4  # base.params["motion"]["max"]["accel_m"]
ACC_ANG = 1.2  # (accel_m_max - accel_m_default) / wheel_separation_m
MAX_HEADING_ANG = np.pi / 4


class DiffDriveVelocityController(abc.ABC):
    """
    Abstract class for differential drive robot velocity controllers.
    """

    @abc.abstractmethod
    def __call__(self, xyt_err) -> Tuple[float, float]:
        pass


class DDVelocityControlNoplan(DiffDriveVelocityController):
    """
    Control logic for differential drive robot velocity control.
    Does not plan at all, instead uses heuristics to gravitate towards the goal.
    """

    def __init__(self, hz):
        self.v_max = V_MAX_DEFAULT
        self.w_max = W_MAX_DEFAULT
        self.lin_error_tol = self.v_max / hz
        self.ang_error_tol = self.w_max / hz

    @staticmethod
    def _velocity_feedback_control(x_err, a, v_max):
        """
        Computes velocity based on distance from target (trapezoidal velocity profile).
        Used for both linear and angular motion.
        """
        t = np.sqrt(2.0 * abs(x_err) / a)  # x_err = (1/2) * a * t^2
        v = min(a * t, v_max)
        return v * np.sign(x_err)

    @staticmethod
    def _turn_rate_limit(lin_err, heading_diff, w_max, tol=0.0):
        """
        Compute velocity limit that prevents path from overshooting goal

        heading error decrease rate > linear error decrease rate
        (w - v * np.sin(phi) / D) / phi > v * np.cos(phi) / D
        v < (w / phi) / (np.sin(phi) / D / phi + np.cos(phi) / D)
        v < w * D / (np.sin(phi) + phi * np.cos(phi))

        (D = linear error, phi = angular error)
        """
        assert lin_err >= 0.0
        assert heading_diff >= 0.0

        if heading_diff > MAX_HEADING_ANG:
            return 0.0
        else:
            return (
                w_max
                * lin_err
                / (np.sin(heading_diff) + heading_diff * np.cos(heading_diff) + 1e-5)
            )

    def __call__(self, xyt_err):
        v_cmd = w_cmd = 0

        lin_err_abs = np.linalg.norm(xyt_err[0:2])
        ang_err = xyt_err[2]

        # Go to goal XY position if not there yet
        if lin_err_abs > self.lin_error_tol:
            heading_err = np.arctan2(xyt_err[1], xyt_err[0])
            heading_err_abs = abs(heading_err)

            # Compute linear velocity
            v_raw = self._velocity_feedback_control(lin_err_abs, ACC_LIN, self.v_max)
            v_limit = self._turn_rate_limit(
                lin_err_abs,
                heading_err_abs,
                self.w_max / 2.0,
                tol=self.lin_error_tol,
            )
            v_cmd = np.clip(v_raw, 0.0, v_limit)

            # Compute angular velocity
            w_cmd = self._velocity_feedback_control(heading_err, ACC_ANG, self.w_max)

        # Rotate to correct yaw if XY position is at goal
        elif abs(ang_err) > self.ang_error_tol:
            # Compute angular velocity
            w_cmd = self._velocity_feedback_control(ang_err, ACC_ANG, self.w_max)

        return v_cmd, w_cmd