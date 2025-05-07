"""TODO."""

from __future__ import absolute_import, annotations, division, print_function

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import jax
import numpy as np
from lsy_models.models import dynamics_numeric, observation_function
from lsy_models.utils.constants import Constants

from lsy_estimators.filterpy import (
    Q_discrete_white_noise,
    ukf_correct,
    ukf_predict,
    ukf_predict_correct,
)
from lsy_estimators.structs import SigmaPointsSettings, UKFData, UKFSettings

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor


class Estimator(ABC):
    """Base class for estimator implementations."""

    def __init__(self, state_dim: int, input_dim: int, obs_dim: int, dt: float):
        """Initialize basic parameters.

        Args:
            state_dim: Dimensionality of the systems states, e.g., x of f(x,u)
            input_dim: Dimensionality of the input to the dynamics, e.g., u of f(x,u)
            obs_dim: Dimensionality of the observations, e.g., y
            dt: Time step between callings.
        """
        self._state_dim = state_dim
        self._obs_dim = obs_dim
        self._input_dim = input_dim
        self._dt = dt
        self._state = np.zeros(self._state_dim)
        self._input = np.zeros(self._input_dim)

    def reset(self):
        """Reset the noise to its initial state."""
        self._state = np.zeros(self._state_dim)
        self._input = np.zeros(self._input_dim)

    @abstractmethod
    def step(self):
        """Increment the noise step for time dependent noise classes."""
        pass

    def set_input(self, u: Array):
        """Sets the input of the dynamical system. Assuming this class gets called multiple times between controller calls. We therefore store the input as a constant in the class.

        Args:
            u: Input to the dynamical system.
        """
        # TODO check for shape?
        self.data = self.data.replace(u=u)


class KalmanFilter(Estimator):
    """Unscented Kalman Filter class that wraps the filterPY toolbox."""

    def __init__(
        self,
        dt: float,
        model: str = "mellinger_rpyt",  # mellinger_rpyt, fitted_DI_rpy
        config: str = "cf2x_L250",
        filter_type: str = "UKF",
        estimate_forces_motor: bool = False,
        estimate_forces_dist: bool = False,
        estimate_torques_dist: bool = False,
        initial_obs: dict[str, Array] | None = None,
    ):  # TODO give obs and info # analytical_mel_att
        """Initialize basic parameters.

        Args:
            dt: Time step between callings.
            model: The name of the model that is to be used.
            config: The setup configuration of the drone.
            filter_type: Either EKF or UKF
            initial_obs: Optional, the initial observation of the environment's state. See the environment's observation space for details.
        """
        fx = dynamics_numeric(model, config)
        hx = observation_function
        # fx = jax.jit(dynamics_numeric(model, config))
        # hx = jax.jit(observation_function)
        self.constants = Constants.from_config(config)

        dim_x = 13
        if estimate_forces_motor:
            dim_x += 4
        if estimate_forces_dist:
            dim_x += 3
        if estimate_torques_dist:
            dim_x += 3
        dim_u = 4  # TODO from available models
        dim_z = 7  # pos, quat

        self.data = UKFData.create_empty(
            forces_motor=estimate_forces_motor,
            forces_dist=estimate_forces_dist,
            torques_dist=estimate_torques_dist,
            dim_u=dim_u,
            dim_z=dim_z,
        )

        dim_x = UKFData.get_state_dim(self.data)
        # print(f"dim_x={dim_x}")

        Q, R = self.create_covariance_matrices(
            dim_x=dim_x,
            dim_z=dim_z,
            varQ_pos=1e-6,
            varQ_quat=1e-2,
            varQ_forces_motor=1e-5,
            varR_pos=1e-9,
            varR_quat=5e-9,
            dt=dt,
        )
        # Q = self.create_Q(
        #     dim_x=dim_x, varQ_pos=1e-6, varQ_quat=1e-6, varQ_vel=1e-3, varQ_ang_vel=1e-2, dt=dt
        # )
        # R = self.create_R(dim_z=dim_z, varR_pos=1e-8, varR_quat=3e-6, dt=dt)

        sigma_settings = SigmaPointsSettings.create(n=dim_x, alpha=1e-3, beta=2.0, kappa=0.0)
        self.settings = UKFSettings.create(SPsettings=sigma_settings, Q=Q, R=R, fx=fx, hx=hx)

        # Initialize state and covariance
        if initial_obs is not None:
            self.set_state(initial_obs["pos"], initial_obs["quat"])

    def set_state(self, pos: Array, quat: Array):
        """Sets pos and quat of the state to the given values and reduces the covariance value.

        With aggressive Q and R values, this needs to be called in the beginning. Otherwise the estimator might fail.
        """
        self.data = self.data.replace(pos=pos, quat=quat)
        dim_x = UKFData.as_state_array(self.data).shape[0]
        self.data = self.data.replace(covariance=np.eye(dim_x) * 1e-6)

    def create_covariance_matrices(
        self,
        dim_x: int,
        dim_z: int,
        varQ_pos: float,
        varQ_quat: float,
        varQ_forces_motor: float,
        varR_pos: float,
        varR_quat: float,
        dt: float,
        varQ_forces_dist: float = 1e-9,
        varQ_torques_dist: float = 1e-12,
    ) -> tuple[Array, Array]:
        """Creates sophisticated covariance matrices based on a paper.

        We assume that the linear elements (pos, vel) are correlated
        and the rotational elements (quat, ang_vel) are.
        For more information consult the filterpy library.
        """
        ### Set process noise covariance (tunable). Uncertainty in the dynamics. High Q -> less trust in model
        Q = np.eye(dim_x)

        # TODO maybe add correlation with forces_motor!

        # Linear equations
        Q_xyz = Q_discrete_white_noise(
            dim=2, dt=dt, var=varQ_pos, block_size=3, order_by_dim=False
        )  # pos & vel
        Q[0:3, 0:3] = Q_xyz[0:3, 0:3]  # pos
        Q[7:10, 7:10] = Q_xyz[3:6, 3:6]  # vel
        Q[0:3, 7:10] = Q_xyz[0:3, 3:6]  # pos <-> vel
        Q[7:10, 0:3] = Q_xyz[3:6, 0:3]  # pos <-> vel

        # Rotational equations
        # since quat and ang_vel have different length, it's a little more tricky
        Q_rot = Q_discrete_white_noise(
            dim=2, dt=dt, var=varQ_quat, block_size=3, order_by_dim=False
        )  # quat & ang_vel
        Q[3:5, 3:5] = Q_rot[0:2, 0:2]  # quat12
        Q[5:7, 5:7] = Q_rot[0:2, 0:2]  # quat34
        Q[10:13, 10:13] = Q_rot[3:6, 3:6]  # ang_vel
        # We know that all quat_dot dependent on all ang_vel
        # => fill in the whole 4x3 and 3x4 matrix blocks with the variance
        Q[3:7, 10:13] = Q_rot[0, 3]  # quat <-> ang_vel
        Q[10:13, 3:7] = Q_rot[3, 0]  # quat <-> ang_vel

        i = 13  # for keeping how big the state (index) is

        # Motor forces TODO: maybe add correlation of acceleration (and angular acc) to motor forces
        if self.data.forces_motor is not None:
            Q_forces = Q_discrete_white_noise(
                dim=2, dt=dt, var=varQ_forces_motor, block_size=1, order_by_dim=False
            )  # Motor Forces
            Q[i : i + 2, i : i + 2] = np.eye(2) * Q_forces[1, 1]  # forces 1&2
            Q[i + 2 : i + 4, i + 2 : i + 4] = np.eye(2) * Q_forces[1, 1]  # forces 3&4
            Q[7:10, i : i + 4] = Q_forces[0, 1]  # forces <-> vel
            Q[i : i + 4, 7:10] = Q_forces[1, 0]  # forces <-> vel
            Q[10:13, i : i + 4] = (
                Q_forces[0, 1] * 0.04
            )  # forces <-> ang_vel, times arm length (F=rxl)
            Q[i : i + 4, 10:13] = (
                Q_forces[1, 0] * 0.04
            )  # forces <-> ang_vel, times arm length (F=rxl)
            # Q[i : i + 4] *= varQ_forces_motor  # TODO move index as in dataclass example and make optional
            i = i + 4

        # External forces and torques TODO Maybe make x and v dependend on F uncertainty and same for torque
        if self.data.forces_dist is not None:  # TODO move index as in dataclass example
            Q[i : i + 3, i : i + 3] *= varQ_forces_dist  # Force
            i = i + 3
        if self.data.torques_dist is not None:
            Q[i : i + 3, i : i + 3] *= varQ_torques_dist  # Torque

        ### Set measurement noise covariance (tunable). Uncertaints in the measurements. High R -> less trust in measurements
        R = np.eye(dim_z)  # Assuming uncorrelated noise
        # very low noise on the position ("mm precision" => even less noise)
        R[:3, :3] = R[:3, :3] * varR_pos
        # "high" measurements noise on the angles, estimate: 0.01 constains all values => std=3e-3 TODO look at new quat measurements
        R[3:, 3:] = R[3:, 3:] * varR_quat

        return Q, R

    def create_Q(
        self,
        dim_x: int,
        varQ_pos: float,
        varQ_quat: float,
        varQ_vel: float,
        varQ_ang_vel: float,
        dt: float,
        varQ_forces_motor: float = 1e-1,
        varQ_forces_dist: float = 1e-9,
        varQ_torques_dist: float = 1e-12,
    ) -> Array:
        Q = np.eye(dim_x)
        # TODO remove, just for testing
        Q[0:3] *= varQ_pos
        Q[3:7] *= varQ_quat
        Q[7:10] *= varQ_vel
        Q[10:13] *= varQ_ang_vel
        i = 13
        if self.data.forces_motor is not None:
            Q[i : i + 4] *= varQ_forces_motor
            i = i + 4
        if self.data.forces_dist is not None:
            Q[i : i + 3, i : i + 3] *= varQ_forces_dist  # Force
            i = i + 3
        if self.data.torques_dist is not None:
            Q[i : i + 3, i : i + 3] *= varQ_torques_dist  # Torque
        return Q

    def create_R(self, dim_z: int, varR_pos: float, varR_quat: float, dt: float) -> Array:
        ### Set measurement noise covariance (tunable). Uncertaints in the measurements. High R -> less trust in measurements
        R = np.eye(dim_z)  # Assuming uncorrelated noise
        # very low noise on the position ("mm precision" => even less noise)
        R[:3, :3] = R[:3, :3] * varR_pos
        # "high" measurements noise on the angles, estimate: 0.01 constains all values => std=3e-3 TODO look at new quat measurements
        R[3:, 3:] = R[3:, 3:] * varR_quat
        return R

    def step(self, pos: Array, quat: Array, dt: float, command: Array | None = None) -> UKFData:
        """Steps the UKF by one. Doing one prediction and correction step.

        Args:
            obs: Latest observation in the form of a dict with "pos" and "rpy"
            dt: Optional, time step size. If not specified, default time is used
            u: Optional, latest input to the system

        Return:
            New state prediction
        """
        # Check if time step is positive
        if dt <= 0:
            return self.data

        # Update the input
        if command is not None:
            self.set_input(command)

        # Update observation and dt
        # dt hast to be vectorized to work properly in jax
        self.data = self.data.replace(z=np.concat((pos, quat)), dt=np.array([dt]))

        # if self.data.dt > 0:  # TODO make dt check more elegant and catch all errors
        # self.data = ukf_predict(self.data, self.settings)
        # self.data = ukf_predict(self.data, self.settings)
        self.data = ukf_predict_correct(self.data, self.settings)

        return self.data

    def predict(self, dt: float, command: Array | None = None) -> UKFData:
        """TODO."""
        # Check if time step is positive
        if dt <= 0:
            return self.data

        # Update the input
        if command is not None:
            self.set_input(command)

        # Update observation and dt
        # dt hast to be vectorized to work properly in jax
        self.data = self.data.replace(dt=np.array([dt]))

        self.data = ukf_predict(self.data, self.settings)

        return self.data

    def correct(self, pos: Array, quat: Array, command: Array | None = None) -> UKFData:
        """TODO."""
        # Update the input
        if command is not None:
            self.set_input(command)

        # Update observation and dt
        # dt hast to be vectorized to work properly in jax
        self.data = self.data.replace(z=np.concat((pos, quat)))

        # print(f"{quat=}, {self.data.quat=}, {self.data.u[-2]=}")

        self.data = ukf_correct(self.data, self.settings)

        return self.data


# test = KalmanFilter(1.0 / 200)

# times = []

# for i in range(1000):
#     print(f"iteration {i}")
#     t1 = time.perf_counter()
#     test.step(np.array([0, 0, 0, 0, 0, 0, 1]))
#     t2 = time.perf_counter()
#     times.append(t2 - t1)

# print(f"Avg = {np.mean(times) * 1000}ms")
