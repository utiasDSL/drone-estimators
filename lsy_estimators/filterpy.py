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

import time
from typing import TYPE_CHECKING, Callable

import numpy as np
from lsy_models.utils import rotation as R
from scipy.linalg import block_diag

from lsy_estimators.datacls import SigmaPointsSettings, UKFData, UKFSettings
from lsy_estimators.integration import integrate_UKFData

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor


def ukf_predict_correct(data: UKFData, settings: UKFSettings) -> UKFData:
    """TODO."""
    xp = data.pos.__array_namespace__()
    #### Predict
    # Calculate sigma pointstf.transform.rotation.x,
    # TODO special sigma points for quaternions!
    sigmas = ukf_calculate_sigma_points(data, settings)

    # Pass sigma points through dynamics
    pos_dot, quat_dot, vel_dot, angvel_dot, forces_motor_dot = settings.fx(
        pos=sigmas.pos,
        quat=sigmas.quat,
        vel=sigmas.vel,
        angvel=sigmas.angvel,
        forces_motor=sigmas.forces_motor,
        forces_dist=sigmas.forces_dist,
        torques_dist=sigmas.torques_dist,
        command=data.u,
    )
    sigmas_dot = UKFData.create(pos_dot, quat_dot, vel_dot, angvel_dot, forces_motor_dot)
    # print(f"derivatives: {data_sigmas_dot.angvel}")
    # sigmas_dot = QuadrotorState.as_array(sigma_states_dot)

    # print(f"function call = {(t2 - t1) * 1000}ms, as_array = {(t3 - t2) * 1000}ms")

    # Integrate dynamics if continuous
    # TODO implement proper integrator
    # TODO watch out for quaternion integration! (length and orientation)
    # sigmas_f = sigmas + sigmas_dot * data.dt
    # sigmas_f[..., 3:7] = (
    #     sigmas_f[..., 3:7] / xp.linalg.norm(sigmas_f[..., 3:7], axis=-1)[:, None]
    # )  # TODO jax cant do that in place
    # data = data.replace(sigmas_f=sigmas_f)
    data_sigmas_f = integrate_UKFData(sigmas, sigmas_dot)
    sigmas_f = UKFData.as_state_array(data_sigmas_f)

    # For the unscented tranform, rotation as vector is needed again
    # It can be computed by forming dquat into drotvec
    sigmas_f_drotvec = UKFData.as_drotvec_array(data_sigmas_f)
    # Compute prior with unscented transform
    x, P = ukf_unscented_transform(
        sigmas_f_drotvec, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.Q
    )
    # Transforming the drotvec representation back to quaterion state
    data_sigmas_f_UT = UKFData.from_drotvec_array(data_sigmas_f, x)
    # sigmas_f_UT = UKFData.as_state_array(data_sigmas_f_UT)

    #### Correct
    # Pass prior sigmas through measurment function h(x,u,dt) to get measurement sigmas
    # sigmas_h = settings.hx(sigmas_f, data.u, data.dt)
    sigmas_h = settings.hx(
        pos=data_sigmas_f.pos,
        quat=data_sigmas_f.quat,
        vel=data_sigmas_f.vel,
        angvel=data_sigmas_f.angvel,
        forces_motor=data_sigmas_f.forces_motor,
        forces_dist=data_sigmas_f.forces_dist,
        torques_dist=data_sigmas_f.torques_dist,
        command=data.u,
    )
    sigmas_h_drotvec = sigmas_f_drotvec[..., :6]  # TODO replace this Ghetto version with hx
    # data = data.replace(sigmas_h=sigmas_h)

    # Pass mean and covariance of prediction through unscented transform
    zp, S = ukf_unscented_transform(
        sigmas_h_drotvec, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.R
    )
    # SI = xp.linalg.inv(data.S)
    # data = data.replace(S=S, SI=SI)
    # data = data.replace(S=S)

    # compute cross variance
    Pxz = ukf_cross_variance(x, zp, sigmas_f_drotvec, sigmas_h_drotvec, settings.SPsettings.Wc)
    # K = xp.dot(Pxz, data.SI)       # Kalman gain
    # K @ S = Pxz => K = Pxz @ S^-1 => or: S.T @ K.T = Pxz.T
    K = xp.linalg.solve(S.T, Pxz.T).T
    # Transforming the measurement into drotvec form
    quat_pred = data_sigmas_f_UT.quat
    quat_meas = data.z[3:]
    drotvec = (R.from_quat(quat_meas) * R.from_quat(quat_pred).inv()).as_rotvec()
    z_drotvec = xp.concat((data.z[:3], drotvec))
    y = xp.subtract(z_drotvec, zp)  # residual
    # data = data.replace(K=K, y=y)

    # Update Gaussian state estimate (x, P)
    x = x + xp.dot(K, y)
    # print(f"P prior = {xp.diag(P)}")
    # print(f"P prior = \n{P}")
    # Added identity for numerical stability
    P = P - xp.dot(K, xp.dot(S, K.T))  # + xp.eye(P.shape[0]) * 1e-9
    # print(f"P post = {xp.diag(P)}")
    # print(f"P post = \n{P}")

    # Save posterior
    data = UKFData.from_drotvec_array(data_sigmas_f, x)
    data = data.replace(covariance=P)

    return data


# Legacy code
# def ukf_predict(data: UKFData, settings: UKFSettings) -> UKFData:
#     """TODO."""
#     # Calculate sigma points
#     sigmas = ukf_calculate_sigma_points(data, settings)
#     sigma_states = QuadrotorState.from_array(data.state, sigmas)

#     # Pass sigma points through dynamics
#     sigma_states_dot = settings.fx(sigma_states, data.u)
#     sigmas_dot = QuadrotorState.as_array(sigma_states_dot)

#     # Integrate dynamics if continuous
#     # TODO implement proper integrator
#     # TODO watch out for quaternion integration!
#     sigmas_f = sigmas + sigmas_dot * data.dt
#     data = data.replace(sigmas_f=sigmas_f)

#     # Compute prior with unscented transform
#     x, P = ukf_unscented_transform(
#         data.sigmas_f, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.Q
#     )

#     # save prior
#     data = data.replace(x=x, covariance=P)

#     return data


# def ukf_correct(data: UKFData, settings: UKFSettings) -> UKFData:
#     """TODO."""
#     xp = data.covariance.__array_namespace__()
#     # Pass prior sigmas through measurment function h(x,u,dt) to get measurement sigmas
#     sigmas_h = settings.hx(data.sigmas_f, data.u, data.dt)
#     data = data.replace(sigmas_h=sigmas_h)

#     # Pass mean and covariance of prediction through unscented transform
#     zp, S = ukf_unscented_transform(
#         data.sigmas_h, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.R
#     )
#     # SI = xp.linalg.inv(data.S)
#     # data = data.replace(S=S, SI=SI)
#     data = data.replace(S=S)

#     # compute cross variance
#     Pxz = ukf_cross_variance(data.x, zp, data.sigmas_f, data.sigmas_h, settings.SPsettings.Wc)
#     # K = xp.dot(Pxz, data.SI)       # Kalman gain
#     # K @ S = Pxz => K = Pxz @ S^-1 => or: S.T @ K.T = Pxz.T
#     K = xp.linalg.solve(S.T, Pxz.T).T
#     y = xp.subtract(data.z, zp)  # residual
#     data = data.replace(K=K, y=y)

#     # Update Gaussian state estimate (x, P)
#     x = data.x + xp.dot(data.K, data.y)
#     P = data.P - xp.dot(data.K, xp.dot(data.S, data.K.T))

#     # Safe posterior
#     data = data.replace(x=x, P=P, x_post=x, P_post=P)

#     return data


def ukf_calculate_sigma_points(data: UKFData, settings: UKFSettings) -> UKFData:
    """TODO."""
    xp = data.pos.__array_namespace__()

    P = data.covariance
    # Adding some very small identity part for numerical stability
    # Note: Higher values make the system more stable for the cost of more noise!
    P = P + xp.eye(P.shape[0]) * 1e-12
    U = xp.linalg.cholesky((settings.SPsettings.lambda_ + settings.SPsettings.n) * P, upper=True)

    # Calculate sigmas based on a delta angle representation with a mean of 0
    state_array_dangle = UKFData.as_drotvec_array(data)
    sigma_center = state_array_dangle
    sigma_pos = xp.subtract(state_array_dangle, -U)
    sigma_neg = xp.subtract(state_array_dangle, U)
    sigmas = xp.vstack((sigma_center, sigma_pos, sigma_neg))

    # Convert delta angles to delta quaternions and then rotate the mean accordingly
    return UKFData.from_drotvec_array(data, sigmas)


def ukf_unscented_transform(
    sigmas: Array, Wm: Array, Wc: Array, noise_cov: Array = None
) -> tuple[Array, Array]:
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


def ukf_cross_variance(x: Array, z: Array, sigmas_f: Array, sigmas_h: Array, Wc: Array) -> Array:
    """Compute cross variance of the state `x` and measurement `z`."""
    xp = x.__array_namespace__()
    # The slicing brings Wc in the correct shape to be broadcast
    # The einsum as set up here takes the outer product of all the stacked vectors
    Pxz = Wc[:, None, None] * xp.einsum(
        "bi,bj->bij", xp.subtract(sigmas_f, x), xp.subtract(sigmas_h, z)
    )
    Pxz = xp.sum(Pxz, axis=0)
    return Pxz


def order_by_derivative(Q: Array, dim: int, block_size: int) -> Array:
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
) -> Array:
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
