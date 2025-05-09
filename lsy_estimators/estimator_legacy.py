"""A simple smoothing estimator.

This estimator is originally from Chris McKinnon.
It was adapted to work with similarly to the new estimators for backwards compatability.
"""

from __future__ import absolute_import, annotations, division, print_function

import math
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
import transforms3d as tf

from lsy_estimators.structs import UKFData

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor

# from lsy_estimators.quaternions import apply_omega_to_quat, global_to_body, omega_from_quat_quat


class StateEstimator(object):
    """Vicon state estimation and filtering.

    Parameters
    ----------
    filter_parameters : sequence of floats
        The 4 tuning parameters for the Kalman filter
    """

    def __init__(self, filter_parameters: tuple):
        """TODO."""
        # Lock for access to state
        # Makes sure the service does not conflict with normal updates
        self.state_access_lock = Lock()

        # Read in the parameters
        (
            self.tau_est_trans,
            self.tau_est_trans_dot,
            self.tau_est_trans_dot_dot,
            self.tau_est_rot,
            self.tau_est_rot_dot,
        ) = filter_parameters

        # Initialize the state variables
        self.pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.acc = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Initialize rotations (quaternions).
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.omega_g = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Initialize measurements
        self.pos_meas = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.vel_meas = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.acc_meas = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.quat_meas = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.omega_g_meas = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Initialze old measurements
        self.pos_old = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.vel_old = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.quat_old = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Correction to align vicon frame with body frame
        self.quat_corr = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Initialize time..at first time step trusts measurements to 100%
        self.time = 0.0
        self.time_meas = 0.0
        self.dt = 0.0

    @property
    def rpy(self):
        """Convert quaternion to roll, pitch, yaw."""
        raise NotImplementedError()
        # return tf.euler_from_quaternion(self.quat)
        # return R.from_quat(self.quat).as_euler("xyz")

    @property
    def omega_b(self) -> Array:
        """Return the body angular velocity."""
        return global_to_body(self.quat, self.omega_g)

    def step(self, pos: Array, quat: Array, dt: float, command: Array | None = None) -> UKFData:
        """This function is not part of the legacy estimator and only for compatability."""
        self.dt = dt
        # the transforms3d library used here uses scalar first, so we need to make it scalar first
        quat = np.roll(quat, 1)
        self.get_new_measurement(pos, quat)
        self.prior_update()
        self.measurement_update()
        # outside, we use scalar last, so we need to roll back
        return UKFData.create(
            self.pos, np.array(np.roll(self.quat, -1)).astype(float), self.vel, self.omega_b
        )

    def predict(self, dt: float, command: Array | None = None) -> UKFData:
        """This function is not part of the legacy estimator and only for compatability."""
        # Since the legacy estimator doesn't inherently support the prediction/correction
        # form of a Kalman filter, we only step the time in the prediction step. In the
        # correction step, we use the accumulated time step size to correct with the
        # correct dt.
        self.dt += dt
        return UKFData.create(
            self.pos, np.array(np.roll(self.quat, -1)).astype(float), self.vel, self.omega_b
        )

    def correct(self, pos: Array, quat: Array, command: Array | None = None) -> UKFData:
        """This function is not part of the legacy estimator and only for compatability."""
        # Since the legacy estimator doesn't inherently support the prediction/correction
        # form of a Kalman filter, we only step the time in the prediction step. In the
        # correction step, we use the accumulated time step size to correct with the
        # correct dt.
        data = self.step(pos, quat, self.dt, command)
        self.dt = 0
        return data

    def set_input(self, command: Array):
        """This function is not part of the legacy estimator and only for compatability."""
        pass

    def set_state(self, pos: Array, quat: Array):
        """This function is not part of the legacy estimator and only for compatability."""
        self.pos = pos
        self.quat = np.roll(quat, 1)

    def get_new_measurement(self, position: Array, quaternion: Array):
        """Get a new measurement of position and orientation.

        Parameters
        ----------
        time: float
            The measurement time in seconds.
        position: ndarray
        quaternion: ndarray
        """
        # Record the time at which the state was determined
        # self.time_meas = time

        # Calculate time difference, update time
        # self.dt = self.time_meas - self.time

        # Get the translational position from VICON
        self.pos_meas = position

        # Get the rotational position in the form of a quaternion
        self.quat_meas = quaternion

        # Apply correction
        # self.quat_meas = tf.quaternion_multiply(self.quat_meas, self.quat_corr)
        self.quat_meas = tf.quaternions.qmult(self.quat_meas, self.quat_corr)

        # Two quaternions for every rotation, make sure we take
        # the one that is consistent with previous measurements
        if np.dot(self.quat_old, self.quat_meas) < 0.0:
            self.quat_meas = -self.quat_meas

        # Don't compute finite difference for impossibly small time differences
        if self.dt <= 1e-15:
            return

        # Numeric derivatives: Comput velocities
        self.vel_meas = (self.pos_meas - self.pos_old) / self.dt

        # Numeric derivatives: Compute accelerations
        self.acc_meas = (self.vel_meas - self.vel_old) / self.dt

        # Numeric derivatives: Compute angular velocity
        self.omega_g_meas = omega_from_quat_quat(self.quat_old, self.quat_meas, self.dt)

        # Update old measurements (make a copy)
        self.pos_old[:] = self.pos_meas
        self.vel_old[:] = self.vel_meas
        self.quat_old[:] = self.quat_meas

    def prior_update(self):
        """Predict future states using a double integrator model."""
        # Acceleration and angular velocity are assumed constant
        # Update position and orientation
        self.pos += self.dt * self.vel + 0.5 * self.dt * self.dt * self.acc
        self.vel += self.dt * self.acc
        self.quat = apply_omega_to_quat(self.quat, self.omega_g, self.dt)

    def measurement_update(self):
        """Update state estimate with measurements."""
        # NOTE: Use raw position and quaternion - no low pass filter
        # Calculate current Kalman filter gains
        c1 = math.exp(-self.dt / self.tau_est_trans)
        c2 = math.exp(-self.dt / self.tau_est_trans_dot)
        c3 = math.exp(-self.dt / self.tau_est_trans_dot_dot)

        d1 = math.exp(-self.dt / self.tau_est_rot)
        d2 = math.exp(-self.dt / self.tau_est_rot_dot)

        # Wait while locked, then lock itself
        with self.state_access_lock:
            # Measurement updates
            self.pos = (1.0 - c1) * self.pos_meas + c1 * self.pos
            self.vel = (1.0 - c2) * self.vel_meas + c2 * self.vel
            self.acc = (1.0 - c3) * self.acc_meas + c3 * self.acc

            self.quat = (1.0 - d1) * self.quat_meas + d1 * self.quat
            self.omega_g = (1.0 - d2) * self.omega_g_meas + d2 * self.omega_g

            # Make sure that numerical errors don't pile up
            # self.quat /= tf.vector_norm(self.quat)
            self.quat /= tf.quaternions.qnorm(self.quat)

            self.time = self.time_meas


def omega_from_quat_quat(q1: Array, q2: Array, dt: float) -> Array:
    """Convert two quaternions and the time difference to angular velocity.

    Parameters:
    -----------
    q1: quaternion
        The old quaternion
    q2: quaternion
        The new quaternion
    dt: float
        The time difference

    Returns:
    --------
    omega_g: ndarray
        The angular velocity in global coordinates
    """
    if tf.quaternions.qnorm(q1 - q2) < 1e-8:
        # linearly interpolate
        # the quaternion does not stay on unit sphere -> only for very small
        # rotations!

        # dq/dt
        dq = (q2 - q1) / dt

        # From Diebel: Representing Atitude, 6.6, quaternions are defined
        # differently there: [w, x, y, z] instead of [x, y, z, w]!
        omega = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        omega[0] = 2.0 * (q2[0] * dq[1] - q2[3] * dq[2] + q2[2] * dq[3] - q2[1] * dq[0])
        omega[1] = 2.0 * (q2[3] * dq[1] + q2[0] * dq[2] - q2[1] * dq[3] - q2[2] * dq[0])
        omega[2] = 2.0 * (-q2[2] * dq[1] + q2[1] * dq[2] + q2[0] * dq[3] - q2[3] * dq[0])

        return omega
    else:
        # This function becomes numerically unstable for q1-q2 --> 0

        # Find rotation from q1 to q2
        # unit quaternion -> conjugate is the same as inverse
        # q2 = r * q1 --> r = q2 * inv(q1)
        r = tf.quaternions.qmult(q2, tf.quaternions.qconjugate(q1))
        # r = tf.quaternion_multiply(q2, tf.quaternion_conjugate(q1))
        r /= tf.quaternions.qnorm(r)

        # Angle of rotation
        # angle = 2.0 * math.acos(r[3])
        angle = 2.0 * math.acos(r[0])

        # acos gives value in [0,pi], ensure that we take the short path
        # (e.g. rotate by -pi/2 rather than 3pi/2)
        if angle > math.pi:
            angle -= 2.0 * math.pi

        # angular velocity = angle / dt
        # axis of rotation corresponds to r[:3]
        # return angle / dt * r[:3] / tf.quaternions.qnorm(r[:3])
        return angle / dt * r[1:] / tf.quaternions.qnorm(r[1:])


def apply_omega_to_quat(q: Array, omega: Array, dt: float) -> Array:
    """Convert a quaternion q and apply the angular velocity omega to it over dt.

    Parameters:
    -----------
    q: quaternion
    omega: ndarray
        angular velocity
    dt: float
        time difference

    Returns:
    --------
    quaternion
        The quaternion of the orientation after rotation with omega for dt
        seconds.
    """
    # rotation angle around each axis
    w = omega * dt

    # only rotate if the angle we rotate through is actually significant
    if tf.quaternions.qnorm(w) < np.finfo(float).eps * 4.0:
        return q

    # quaternion corresponding to this rotation
    # w = 0 is not a problem because numpy is awesome
    r = tf.quaternions.axangle2quat(w, tf.quaternions.qnorm(w))
    # r = tf.quaternion_about_axis(np.linalg.norm(w), w)

    # return the rotated quaternion closest to original
    return tf.quaternions.qmult(r, q)


def global_to_body(q: Array, vec: Array) -> Array:
    """Convert a vector from global to body coordinates.

    Parameters:
    -----------
    q: quaternion
        The rotation quaternion
    vec: ndarray
        The vector in global coordinates

    Returns:
    vec: ndarray
        The vector in body coordinates
    """
    # tf.quaternion_matrix(q)[:3,:3] is a homogenous rotation matrix that
    # rotates a vector by q
    # tf.quaternion_matrix(q)[:3,:3] is rot. matrix from body to global frame
    # its transpose is the trafo matrix from global to body
    # that matrix is multiplied by omega
    return np.dot(tf.quaternions.quat2mat(q).transpose(), vec)


def body_to_global(q: Array, vec: Array) -> Array:
    """Convert a vector from global to body coordinates.

    Parameters:
    -----------
    q: quaternion
        The rotation quaternion
    vec: ndarray
        The vector in body coordinates

    Returns:
    vec: ndarray
        The vector in global coordinates
    """
    # tf.quaternion_matrix(q)[:3,:3] is a homogenous rotation matrix that
    # rotates a vector by q
    # tf.quaternion_matrix(q)[:3,:3] is the matrix from body to global frame
    # that matrix is multiplied by omega
    return np.dot(tf.quaternions.quat2mat(q), vec)
    # return np.dot(tf.quaternion_matrix(q)[:3, :3], vec)
