"""This file contains modified methods and classes taken from the filterpy library.

Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import absolute_import, annotations, division, print_function

from typing import TYPE_CHECKING

import jax
import numpy as np
from scipy.linalg import block_diag

from lsy_estimators.integration import integrate_UKFData
from lsy_estimators.structs import UKFData, UKFSettings

if TYPE_CHECKING:
    from numpy.typing import NDArray


# @jax.jit
def ukf_predict_correct(data: UKFData, settings: UKFSettings) -> UKFData:
    """TODO."""
    xp = data.pos.__array_namespace__()
    #### Predict
    # Calculate sigma points
    # TODO special sigma points for quaternions!
    sigmas = ukf_calculate_sigma_points(data, settings)
    data_sigmas = UKFData.from_state_array(data, sigmas)

    # Pass sigma points through dynamics
    pos_dot, quat_dot, vel_dot, ang_vel_dot, forces_motor_dot = settings.fx(
        pos=data_sigmas.pos,
        quat=data_sigmas.quat,
        vel=data_sigmas.vel,
        ang_vel=data_sigmas.ang_vel,
        forces_motor=data_sigmas.forces_motor,
        forces_dist=data_sigmas.forces_dist,
        torques_dist=data_sigmas.torques_dist,
        command=data.u,
    )
    data_sigmas_dot = UKFData.create(pos_dot, quat_dot, vel_dot, ang_vel_dot, forces_motor_dot)

    # Integrate dynamics if continuous
    data_sigmas_f = integrate_UKFData(data_sigmas, data_sigmas_dot)
    sigmas_f = UKFData.as_state_array(data_sigmas_f)
    data = data.replace(sigmas_f=sigmas_f)

    # Pass prior sigmas through measurment function h(x,u,dt) to get measurement sigmas
    sigmas_h = settings.hx(
        pos=data_sigmas_f.pos,
        quat=data_sigmas_f.quat,
        vel=data_sigmas_f.vel,
        ang_vel=data_sigmas_f.ang_vel,
        command=data.u,
        forces_motor=data_sigmas_f.forces_motor,
        forces_dist=data_sigmas_f.forces_dist,
        torques_dist=data_sigmas_f.torques_dist,
    )
    data = data.replace(sigmas_h=sigmas_h)

    # Compute prior with unscented transform
    x, P = ukf_unscented_transform(
        sigmas_f, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.Q
    )

    # save prior
    data = UKFData.from_state_array(data, x)
    data = data.replace(covariance=P)

    #### Correct
    # Pass mean and covariance of prediction through unscented transform
    zp, S = ukf_unscented_transform(
        data.sigmas_h, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.R
    )
    # SI = xp.linalg.inv(data.S)
    # data = data.replace(S=S, SI=SI)
    # data = data.replace(S=S)

    # compute cross variance
    Pxz = ukf_cross_variance(
        UKFData.as_state_array(data), zp, data.sigmas_f, data.sigmas_h, settings.SPsettings.Wc
    )
    # K = xp.dot(Pxz, data.SI)       # Kalman gain
    # K @ S = Pxz => K = Pxz @ S^-1 => or: S.T @ K.T = Pxz.T
    K = xp.linalg.solve(S.T, Pxz.T).T
    y = xp.subtract(data.z, zp)  # residual
    # data = data.replace(K=K, y=y)

    # Update Gaussian state estimate (x, P)
    x = UKFData.as_state_array(data) + xp.dot(K, y)
    # print(f"P prior = {xp.diag(P)}")
    # print(f"P prior = \n{P}")
    # Added identity for numerical stability
    P = data.covariance - xp.dot(K, xp.dot(S, K.T))  # + xp.eye(P.shape[0]) * 1e-9
    # print(f"P post = {xp.diag(P)}")
    # print(f"P post = \n{P}")

    # Save posterior
    data = UKFData.from_state_array(data, x)
    data = data.replace(covariance=P)

    return data


# @jax.jit
def ukf_predict(data: UKFData, settings: UKFSettings) -> UKFData:
    """TODO."""
    xp = data.pos.__array_namespace__()
    #### Predict
    # Calculate sigma pointstf.transform.rotation.x,
    # TODO special sigma points for quaternions!
    sigmas = ukf_calculate_sigma_points(data, settings)
    data_sigmas = UKFData.from_state_array(data, sigmas)

    # Pass sigma points through dynamics
    pos_dot, quat_dot, vel_dot, ang_vel_dot, forces_motor_dot = settings.fx(
        pos=data_sigmas.pos,
        quat=data_sigmas.quat,
        vel=data_sigmas.vel,
        ang_vel=data_sigmas.ang_vel,
        forces_motor=data_sigmas.forces_motor,
        forces_dist=data_sigmas.forces_dist,
        torques_dist=data_sigmas.torques_dist,
        command=data.u,
    )
    # print(f"{pos_dot=}")
    data_sigmas_dot = UKFData.create(pos_dot, quat_dot, vel_dot, ang_vel_dot, forces_motor_dot)

    # Integrate dynamics if continuous
    data_sigmas_f = integrate_UKFData(data_sigmas, data_sigmas_dot)
    sigmas_f = UKFData.as_state_array(data_sigmas_f)
    data = data.replace(sigmas_f=sigmas_f)

    # Pass prior sigmas through measurment function h(x,u,dt) to get measurement sigmas
    sigmas_h = settings.hx(
        pos=data_sigmas_f.pos,
        quat=data_sigmas_f.quat,
        vel=data_sigmas_f.vel,
        ang_vel=data_sigmas_f.ang_vel,
        command=data.u,
        forces_motor=data_sigmas_f.forces_motor,
        forces_dist=data_sigmas_f.forces_dist,
        torques_dist=data_sigmas_f.torques_dist,
    )
    data = data.replace(sigmas_h=sigmas_h)

    # Compute prior with unscented transform
    x, P = ukf_unscented_transform(
        sigmas_f, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.Q
    )

    # save prior
    data = UKFData.from_state_array(data, x)
    data = data.replace(covariance=P)

    return data


# @jax.jit
def ukf_correct(data: UKFData, settings: UKFSettings) -> UKFData:
    """TODO."""
    xp = data.covariance.__array_namespace__()
    # Pass mean and covariance of prediction through unscented transform
    zp, S = ukf_unscented_transform(
        data.sigmas_h, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.R
    )
    # SI = xp.linalg.inv(data.S)
    # data = data.replace(S=S, SI=SI)
    # data = data.replace(S=S)

    # compute cross variance
    Pxz = ukf_cross_variance(
        UKFData.as_state_array(data), zp, data.sigmas_f, data.sigmas_h, settings.SPsettings.Wc
    )
    # K = xp.dot(Pxz, data.SI)       # Kalman gain
    # K @ S = Pxz => K = Pxz @ S^-1 => or: S.T @ K.T = Pxz.T
    K = xp.linalg.solve(S.T, Pxz.T).T
    y = xp.subtract(data.z, zp)  # residual
    # data = data.replace(K=K, y=y)

    # Update Gaussian state estimate (x, P)
    x = UKFData.as_state_array(data) + xp.dot(K, y)
    # print(f"P prior = {xp.diag(P)}")
    # print(f"P prior = \n{P}")
    # Added identity for numerical stability
    P = data.covariance - xp.dot(K, xp.dot(S, K.T))  # + xp.eye(P.shape[0]) * 1e-9
    # print(f"P post = {xp.diag(P)}")
    # print(f"P post = \n{P}")

    # Save posterior
    data = UKFData.from_state_array(data, x)
    data = data.replace(covariance=P)

    return data


def ukf_calculate_sigma_points(data: UKFData, settings: UKFSettings) -> NDArray[np.floating]:
    """TODO."""
    xp = data.pos.__array_namespace__()
    P = data.covariance
    # Adding some very small identity part for numerical stability
    # Note: Higher values make the system more stable for the cost of more noise!
    P = P + xp.eye(P.shape[0]) * 1e-12
    U = xp.linalg.cholesky((settings.SPsettings.lambda_ + settings.SPsettings.n) * P, upper=True)

    state_array = UKFData.as_state_array(data)
    sigma_center = state_array
    # TODO for the quaternions use more sophisitcated approach to keep length 1!
    # Mainly: rotate in tangent space
    sigma_pos = xp.subtract(state_array, -U)
    sigma_neg = xp.subtract(state_array, U)
    return xp.vstack((sigma_center, sigma_pos, sigma_neg))


def ukf_unscented_transform(
    sigmas: NDArray[np.floating],
    Wm: NDArray[np.floating],
    Wc: NDArray[np.floating],
    noise_cov: NDArray[np.floating] = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """TODO."""
    xp = sigmas.__array_namespace__()
    x = xp.dot(Wm, sigmas)

    # new covariance is the sum of the outer product
    # of the residuals times the weights
    y = sigmas - x[None, :]
    P = xp.dot(y.T, xp.dot(xp.diag(Wc), y))

    if noise_cov is not None:
        P = P + noise_cov

    return (x, P)


def ukf_cross_variance(
    x: NDArray[np.floating],
    z: NDArray[np.floating],
    sigmas_f: NDArray[np.floating],
    sigmas_h: NDArray[np.floating],
    Wc: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute cross variance of the state `x` and measurement `z`."""
    xp = x.__array_namespace__()
    # The slicing brings Wc in the correct shape to be broadcast
    # The einsum as set up here takes the outer product of all the stacked vectors
    Pxz = Wc[:, None, None] * xp.einsum(
        "bi,bj->bij", xp.subtract(sigmas_f, x), xp.subtract(sigmas_h, z)
    )
    Pxz = xp.sum(Pxz, axis=0)
    return Pxz


def order_by_derivative(Q: NDArray[np.floating], dim: int, block_size: int) -> NDArray[np.floating]:
    """TODO."""
    xp = Q.__array_namespace__()

    N = dim * block_size

    D = xp.zeros((N, N))

    Q = xp.array(Q)
    for i, x in enumerate(Q.ravel()):
        f = xp.eye(block_size) * x

        ix, iy = (i // dim) * block_size, (i % dim) * block_size
        D[ix : ix + block_size, iy : iy + block_size] = f

    return D


def Q_discrete_white_noise(
    dim: int, dt: float = 1.0, var: float = 1.0, block_size: int = 1, order_by_dim: bool = True
) -> NDArray[np.floating]:
    """TODO."""
    if not (dim == 2 or dim == 3 or dim == 4):
        raise ValueError("dim must be between 2 and 4")

    if dim == 2:
        Q = [[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]]
    elif dim == 3:
        Q = [
            [0.25 * dt**4, 0.5 * dt**3, 0.5 * dt**2],
            [0.5 * dt**3, dt**2, dt],
            [0.5 * dt**2, dt, 1],
        ]
    else:
        Q = [
            [(dt**6) / 36, (dt**5) / 12, (dt**4) / 6, (dt**3) / 6],
            [(dt**5) / 12, (dt**4) / 4, (dt**3) / 2, (dt**2) / 2],
            [(dt**4) / 6, (dt**3) / 2, dt**2, dt],
            [(dt**3) / 6, (dt**2) / 2, dt, 1.0],
        ]

    if order_by_dim:
        return block_diag(*[Q] * block_size) * var
    return order_by_derivative(np.array(Q), dim, block_size) * var
