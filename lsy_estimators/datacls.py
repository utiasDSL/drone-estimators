"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from flax.struct import Callable, dataclass
from lsy_models.utils import rotation as R

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor


@dataclass
class UKFData:
    """TODO."""

    pos: Array
    quat: Array
    vel: Array
    angvel: Array
    forces_motor: Array | None
    forces_dist: Array | None
    torques_dist: Array | None
    covariance: Array  # Covariance matrix

    u: Array  # input
    z: Array  # measurement
    dt: float

    @classmethod
    def create_empty(
        cls,
        forces_motor: bool = False,
        forces_dist: bool = False,
        torques_dist: bool = False,
        dim_u: int = 4,
        dim_z: int = 7,
    ) -> UKFData:
        """TODO."""
        pos = np.zeros(3)
        quat = np.array([0, 0, 0, 1])
        vel = np.zeros(3)
        angvel = np.zeros(3)
        dim_x = 12  # not 13 because rotation as rotvec
        if forces_motor:
            forces_motor = np.zeros(4)
            dim_x = dim_x + 4
        else:
            forces_motor = None
        if forces_dist:
            forces_dist = np.zeros(3)
            dim_x = dim_x + 3
        else:
            forces_dist = None
        if torques_dist:
            torques_dist = np.zeros(3)
            dim_x = dim_x + 3
        else:
            torques_dist = None

        covariance = np.eye(dim_x)

        u = np.zeros(dim_u)  # input
        z = np.zeros(dim_z)  # measurement
        dt = 1

        return cls(
            pos, quat, vel, angvel, forces_motor, forces_dist, torques_dist, covariance, u, z, dt
        )

    @classmethod
    def create(
        cls,
        pos: Array,
        quat: Array,
        vel: Array,
        angvel: Array,
        forces_motor: Array | None = None,
        forces_dist: Array | None = None,
        torques_dist: Array | None = None,
    ) -> UKFData:
        """TODO."""
        dim_x = 12  # not 13 because rotation as rotvec
        if forces_motor is None:
            dim_x = dim_x + 4
        if forces_dist is None:
            dim_x = dim_x + 3
        if torques_dist is None:
            dim_x = dim_x + 3

        covariance = np.eye(dim_x)

        u = np.zeros(4)  # input
        z = np.zeros(7)  # measurement
        dt = 1

        return cls(
            pos, quat, vel, angvel, forces_motor, forces_dist, torques_dist, covariance, u, z, dt
        )

    @classmethod
    def as_state_array(cls, data: UKFData) -> Array:
        """Returns the state as an array."""
        xp = data.pos.__array_namespace__()
        x = xp.concat((data.pos, data.quat, data.vel, data.angvel), axis=-1)
        if data.forces_motor is not None:
            x = xp.concat((x, data.forces_motor), axis=-1)
        if data.forces_dist is not None:
            x = xp.concat((x, data.forces_dist), axis=-1)
        if data.torques_dist is not None:
            x = xp.concat((x, data.torques_dist), axis=-1)
        return x

    @classmethod
    def from_state_array(cls, data: UKFData, array: Array) -> UKFData:
        """Updates data in the given structure based on a given array."""
        pos = array[..., 0:3]
        quat = array[..., 3:7]
        vel = array[..., 7:10]
        angvel = array[..., 10:13]
        idx = 13
        if data.forces_motor is not None:
            forces_motor = array[..., idx : idx + 4]
            idx = idx + 4
        else:
            forces_motor = None
        if data.forces_dist is not None:
            forces_dist = array[..., idx : idx + 3]
            idx = idx + 3
        else:
            forces_dist = None
        if data.torques_dist is not None:
            torques_dist = array[..., idx : idx + 3]
            idx = idx + 3
        else:
            torques_dist = None

        return data.replace(
            pos=pos,
            quat=quat,
            vel=vel,
            angvel=angvel,
            forces_motor=forces_motor,
            forces_dist=forces_dist,
            torques_dist=torques_dist,
        )

    # @classmethod
    # def as_rotvec_array(cls, data: UKFData) -> Array:
    #     """Returns the state as an array with the orientation as a rotation vector array."""
    #     xp = data.pos.__array_namespace__()
    #     rotvec = R.from_quat(data.quat).as_rotvec()
    #     x = xp.concat((data.pos, rotvec, data.vel, data.angvel), axis=-1)
    #     if data.forces_motor is not None:
    #         x = xp.concat((x, data.forces_motor), axis=-1)
    #     if data.forces_dist is not None:
    #         x = xp.concat((x, data.forces_dist), axis=-1)
    #     if data.torques_dist is not None:
    #         x = xp.concat((x, data.torques_dist), axis=-1)
    #     return x

    @classmethod
    def as_drotvec_array(cls, data: UKFData) -> Array:
        """Returns the state as an array with the orientation as a delta rotation vector."""
        xp = data.pos.__array_namespace__()
        if len(data.quat.shape) == 2:
            drot = R.from_quat(data.quat) * R.from_quat(data.quat[0]).inv()
            drotvec = drot.as_rotvec()
        else:
            drotvec = xp.zeros(3)
        x = xp.concat((data.pos, drotvec, data.vel, data.angvel), axis=-1)
        if data.forces_motor is not None:
            x = xp.concat((x, data.forces_motor), axis=-1)
        if data.forces_dist is not None:
            x = xp.concat((x, data.forces_dist), axis=-1)
        if data.torques_dist is not None:
            x = xp.concat((x, data.torques_dist), axis=-1)
        return x

    @classmethod
    def from_drotvec_array(cls, data: UKFData, array: Array) -> UKFData:
        """Updates data in the given structure based on a given delta roation vector."""
        pos = array[..., 0:3]
        drotvec = array[..., 3:6]
        drot = R.from_rotvec(drotvec)
        if len(drotvec.shape) < len(data.quat.shape):
            rot = R.from_quat(data.quat[0])
        else:
            rot = R.from_quat(data.quat)
        quat = (drot * rot).as_quat()
        vel = array[..., 6:9]
        angvel = array[..., 9:12]
        idx = 12
        if data.forces_motor is not None:
            forces_motor = array[..., idx : idx + 4]
            idx = idx + 4
        else:
            forces_motor = None
        if data.forces_dist is not None:
            forces_dist = array[..., idx : idx + 3]
            idx = idx + 3
        else:
            forces_dist = None
        if data.torques_dist is not None:
            torques_dist = array[..., idx : idx + 3]
            idx = idx + 3
        else:
            torques_dist = None

        return data.replace(
            pos=pos,
            quat=quat,
            vel=vel,
            angvel=angvel,
            forces_motor=forces_motor,
            forces_dist=forces_dist,
            torques_dist=torques_dist,
        )

    @classmethod
    def get_state_dim(cls, data: UKFData) -> int:
        """Returns the dimension of the state."""
        dim_x = 13
        if data.forces_motor is not None:
            dim_x = dim_x + 4
        if data.forces_dist is not None:
            dim_x = dim_x + 3
        if data.torques_dist is not None:
            dim_x = dim_x + 3
        return dim_x


@dataclass
class UKFSettings:
    """TODO."""

    SPsettings: SigmaPointsSettings
    Q: Array
    R: Array
    fx: Callable[
        [Array, Array, Array, Array, Array, Array, Array | None, Array | None],
        tuple[Array, Array, Array, Array, Array | None],
    ]
    hx: Callable[[Array, Array, Array, Array, Array, Array, Array | None, Array | None], Array]

    @classmethod
    def create(
        cls,
        SPsettings: SigmaPointsSettings,
        Q: Array,
        R: Array,
        fx: Callable[
            [Array, Array, Array, Array, Array, Array, Array | None, Array | None],
            tuple[Array, Array, Array, Array, Array | None],
        ],
        hx: Callable[[Array, Array, Array, Array, Array, Array, Array | None, Array | None], Array],
    ) -> UKFSettings:
        """TODO."""
        return cls(SPsettings, Q, R, fx, hx)


@dataclass
class SigmaPointsSettings:
    """TODO."""

    n: int
    alpha: float
    beta: float
    kappa: float
    lambda_: float
    Wc: Array
    Wm: Array

    @classmethod
    def create(cls, n: int, alpha: float, beta: float, kappa: float = 0.0) -> SigmaPointsSettings:
        """TODO."""
        lambda_ = alpha**2 * (n + kappa) - n
        c = 0.5 / (n + lambda_)
        Wc0 = np.array([lambda_ / (n + lambda_) + (1 - alpha**2 + beta)])
        Wm0 = np.array([lambda_ / (n + lambda_)])
        Wc = np.full(2 * n, c)
        Wm = np.full(2 * n, c)
        Wc = np.concat((Wc0, Wc))
        Wm = np.concat((Wm0, Wm))

        return cls(n, alpha, beta, kappa, lambda_, Wc, Wm)
