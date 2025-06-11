"""TODO."""

from __future__ import absolute_import, annotations, division, print_function

from typing import TYPE_CHECKING

import lsy_models.utils.rotation as R
import numpy as np
from lsy_models.models import dynamics_numeric
from lsy_models.utils.constants import Constants

from lsy_estimators.estimator import Estimator

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor


class FxTDO(Estimator):
    """Fixed time Disturbance Observer (FxTDO) as implemented by one of the two publications mentioned below."""

    def __init__(
        self,
        dt: float,
        model: str = "fitted_DI_rpyt",
        config: str = "cf2x_L250",
        estimate_forces_motor: bool = False,
        initial_obs: dict[str, Array] | None = None,
    ):
        """Initialize basic parameters.

        Args:
            dt: Time step between callings.
            model: The name of the model that is to be used.
            config: The setup configuration of the drone.
            estimate_forces_motor: If the motor forces should be estimated, defaults to False.
            initial_obs: Optional, the initial observation of the environment's state. See the environment's observation space for details.
        """
        super().__init__(6, 4, 12, dt)

        self._constants = Constants.from_config(config)
        dynamics = dynamics_numeric(model, config)

        def fx_reduced(
            quat: Array, vel: Array, forces_motor: Array, forces_dist: Array, command: Array
        ) -> Array:
            _, _, vel_dot, _, forces_motor_dot = dynamics(
                pos=np.zeros(3),
                quat=quat,
                vel=vel,
                ang_vel=np.zeros(3),
                command=command,
                constants=self._constants,
                forces_motor=forces_motor,
                forces_dist=forces_dist,
                torques_dist=None,
            )
            return vel_dot, forces_motor_dot

        self._fx = fx_reduced

        # States
        self._vel_hat = np.zeros(3)
        self._forces_dist_hat = np.zeros(3)
        if estimate_forces_motor:
            self._forces_motor_hat = np.zeros(4)
        else:
            self._forces_motor_hat = None

        # Initialize states
        if initial_obs is not None:
            self._vel = initial_obs["vel"]

        # State bounds
        self._delta = 0.34  # f_max [N], no stronger force than gravity (~34g)
        self._delta_bar = 0.001  # f_dot_max [N/s] # TODO tune properly for bound
        self._v_max = 5  # [m/s]
        self._v_dot_max = 10  # [m/s/s] (cant really accelerate faster than free fall)

        # FxTDO, implementation as in
        # "Fixed-time Disturbance Observer-Based MPC Robust Trajectory Tracking Control of Quadrotor" (2024)
        # https://arxiv.org/html/2408.15019v2
        # Hyperparameters:
        self._L1 = 0.1  # Paper: 1.0, observer gain for the observer states
        self._L2 = 0.15  # Paper: 1.0, observer gain for (linear) convergence speed of f_hat_dot, L2 > delta_bar / k2[0]
        self._k1 = np.array([2.0, 0.6, 3.0])  # Paper: [2.0, 0.6, 3.0]
        self._k2 = np.array([2.0, 30.6, 300.0])  # Paper: [2.0, 0.6, 3.0]
        self._d_inf = 0.3  # Paper: 1/3
        # Fixed parameters:
        self._alpha1 = np.array([0.5, 1.0, 1 / (1 - self._d_inf)])
        self._alpha2 = np.array([0.0, 1.0, (1 + self._d_inf) / (1 - self._d_inf)])

        assert self.check_parameters(
            self._delta, self._delta_bar, self._L1, self._L2, self._k1, self._k2, self._d_inf
        ), "Some hyperparameters are not tuned properly"

    def set_parameters(
        self,
        delta: np.floating,
        delta_bar: np.floating,
        L1: np.floating,
        L2: np.floating,
        k1: NDArray[np.floating],
        k2: NDArray[np.floating],
        d_inf: np.floating,
    ):
        """Stores the parameters if valid."""
        if self.check_parameters(delta, delta_bar, L1, L2, k1, k2, d_inf):
            self._delta = delta
            self._delta_bar = delta_bar
            self._L1 = L1
            self._L2 = L2
            self._k1 = k1
            self._k2 = k2
            self._d_inf

    def check_parameters(
        self,
        f_d_max: np.floating,
        f_d_dot_max: np.floating,
        L1: np.floating,
        L2: np.floating,
        k1: NDArray[np.floating],
        k2: NDArray[np.floating],
        d_inf: np.floating,
    ) -> bool:
        """Checks ther parameters for validity. This is only needed to guarantee an upper bound on the estimation time.

        Returns:
            If the parameters are valid.
        """
        # first, simple checks, mainly for a valid observer
        if (
            f_d_max > 0
            and f_d_dot_max > 0
            and L1 > 0
            and L2 > 0
            and np.all(k1 > 0)
            and np.all(k2 > 0)
            and 0 < d_inf < 1
        ):
            # now, more complicated checks, mainly for guarantees
            # print(f"{L2=} / {f_d_dot_max / k2[0]=}")
            if L2 > f_d_dot_max / k2[0]:
                return True
            # TODO X check (see paper)
        else:
            return False

    def step(self, quat: Array, vel: Array, dt: float) -> Array:
        """Steps the observer to calculate the next state and force estimate."""
        if dt <= 0:
            return np.zeros(3)

        self._dt = dt
        e1 = vel - self._vel_hat

        # Calculate derivatives
        vel_hat_dot, force_motor_hat_dot = self._fx(
            quat, self._vel_hat, self._forces_motor_hat, self._forces_dist_hat, self._input
        )
        if self._forces_motor_hat is not None:
            f_t = np.sum(self._forces_motor_hat, axis=-1)
        else:
            f_t = self._input[-1]
        # print(f"{f_t=}")
        z_axis = R.from_quat(quat).as_matrix()[..., -1]
        print(f"fx {vel_hat_dot=}")
        print(f"{z_axis=}")
        vel_hat_dot = (
            1 / self._constants.MASS * z_axis * f_t
            + self._constants.GRAVITY_VEC
            + 1 / self._constants.MASS * self._constants.DI_DD_ACC[2] * self._vel_hat
            + 1
            / self._constants.MASS
            * self._constants.DI_DD_ACC[3]
            * self._vel_hat
            * np.abs(self._vel_hat)
            + 1 / self._constants.MASS * self._forces_dist_hat
            + 1 / self._constants.MASS * self._L1 * self._phi1(e1)
        )
        print(f"{vel_hat_dot=}")
        if self._forces_motor_hat is not None:
            force_motor_hat_dot = (
                1
                / self._constants.DI_D_ACC[2]
                * (self._input[-1][..., None] / 4 - self._forces_motor_hat)
            )
        forces_dist_hat_dot = self._L2 * self._phi2(e1)
        # v_hat_dot = np.clip(v_hat_dot, -self._v_dot_max, self._v_dot_max)  # Clipping
        # f_hat_dot = np.clip(f_hat_dot, -self._delta_bar, self._delta_bar)  # Clipping

        # Integration step (forward Euler)
        self._vel_hat = self._vel_hat + vel_hat_dot * self._dt
        if self._forces_motor_hat is not None:
            self._forces_motor_hat = self._forces_motor_hat + force_motor_hat_dot * self._dt
        self._forces_dist_hat = self._forces_dist_hat + forces_dist_hat_dot * self._dt
        # v_hat = np.clip(v_hat, -self._v_max, self._v_max)  # Clipping
        # f_hat = np.clip(f_hat, -self._delta, self._delta)  # Clipping

        # Storing in the state
        self._state[:3] = self._vel_hat
        self._state[3:] = self._forces_dist_hat

        # return self._state
        return self._forces_dist_hat

    def _phi1(self, e: np.floating) -> np.floating:
        s = 0
        for i in range(3):
            s = s + self._k1[i] * self._ud(e, self._alpha1[i])
        return s

    def _phi2(self, e: np.floating) -> np.floating:
        s = 0
        for i in range(3):
            s = s + self._k2[i] * self._ud(e, self._alpha2[i])
        return s

    def _ud(self, x: NDArray[np.floating], alpha: np.floating) -> NDArray[np.floating]:
        return np.sign(x) * (np.abs(x) ** alpha)
