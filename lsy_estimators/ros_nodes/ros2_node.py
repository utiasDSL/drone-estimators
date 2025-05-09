"""This file contains the estimator node for ROS 2.

Essentially this file provides a wrapper for the estimators to be used with ROS data. The estimates get published to ROS as well.

TODO Subscribed and published topics...
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import pickle
import signal
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING

# Since motion_capture_tracking might be running on multiple PCs simultaneously,
# the same /tf topics get published multiple times to the network. This is
# unintended behavior. Setting it to LOCALHOST will block publishing to the
# network. We also set it here in the estimators s.t. they also don't publish
# to the network. Usually, the estimators get started after
# motion_capture_tracking, but in case the estimators get started first, we
# get consistent behavior.
# Note: All ros nodes started after this line will also only publish locally!
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

import numpy as np
import rclpy
import toml

# Message types: https://docs.ros2.org/foxy/api/geometry_msgs/index-msg.html
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from lsy_models.utils import cf2
from munch import Munch, munchify
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray

from lsy_estimators.estimator import KalmanFilter
from lsy_estimators.estimator_legacy import StateEstimator
from lsy_estimators.ros_nodes.ros2_utils import (
    append_measurement,
    append_state,
    create_array,
    create_marker_array,
    create_pose,
    create_twist,
    create_wrench,
    find_transform,
    tf2array,
)

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import SynchronizedArray
    from multiprocessing.synchronize import Barrier, Event


class MPEstimator:
    """This class starts an estimator and the necessary subscribers and publishers.

    Note: Since all results are published individually, one might want to
    synchronize on the receiving end. For that see
    https://docs.ros.org/en/rolling/p/flex_sync/
    https://github.com/ros2/message_filters

    In the future, a custom message type might be advantageous. However,
    to keep it simple, we use standard message types.
    """

    def __init__(self, settings: Munch):
        """TODO."""
        ctx = mp.get_context("spawn")
        self._shutdown = ctx.Event()
        self._publish_update = ctx.Event()
        startup = ctx.Barrier(3)  # Main process, _subscriber_loop, _publisher_loop

        # Logger setup
        self.logger = logging.getLogger("ESTIMATOR" + "_" + settings.drone_name)
        self.logger.setLevel(logging.INFO)
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # Create formatter with [name][LEVEL] prefix
        formatter = logging.Formatter("[%(name)s][%(levelname)s] %(message)s")
        # Add formatter to handler
        console_handler.setFormatter(formatter)
        # Add handler to logger (only once)
        if not self.logger.hasHandlers():
            self.logger.addHandler(console_handler)
        # We allocate a buffer that is used to store the most recent tf message from the tf
        # subscriber. The message contains four fields: The first element is the number of messages
        # since the array was last cleared. This allows us to throw warnings in case the estimator
        # loop cannot keep up. The second field is the timestamp in s, the third is the position
        # (3d) and the fourth is the quaternion (4d)
        self._tf_msg_buffer = ctx.Array("d", [0.0] * (1 + 1 + 3 + 4))
        # Allocate the command subscriber buffer
        cmd_dim = 4  # May change in the future to be dynamic
        self._cmd_msg_buffer = ctx.Array("d", [0.0] * (1 + 1 + cmd_dim))
        # Estimated state publisher buffers
        self._pose_buffer = ctx.Array("d", [0.0] * 7)
        self._twist_buffer = ctx.Array("d", [0.0] * 6)
        self._forces_buffer = ctx.Array("d", [0.0] * 4)  # Motor forces, optional
        self._wrench_buffer = ctx.Array("d", [0.0] * 6)  # External wrench, optional

        args = (
            settings.drone_name,
            self._tf_msg_buffer,
            self._cmd_msg_buffer,
            startup,
            self._shutdown,
        )
        self._sub_process = ctx.Process(target=self._subscriber_loop, args=args)

        args = (
            settings.drone_name,
            self._pose_buffer,
            self._twist_buffer,
            self._forces_buffer,
            self._wrench_buffer,
            self._publish_update,
            startup,
            self._shutdown,
        )
        self._pub_process = ctx.Process(target=self._publisher_loop, args=args)

        self._sub_process.start()
        self._pub_process.start()
        startup.wait(10.0)

        self.input_needed = False
        self.initial_observation = None
        self.settings = settings
        self.time_stamp_last_prediction = 0
        self.time_stamp_last_correction = 0
        self.perf_timings = deque(maxlen=5000)

        self.frequency = settings.frequency  # Hz # TODO get from vicon frequency

        self.current_header = None
        self.current_state = None

        # This is for storing the data (DEBUG_SAVE_DATA)
        self.data_meas = defaultdict(list)  # {"time": [], "pos": [], "quat": [], "command": []}
        self.data_est = defaultdict(list)

        match self.settings.estimator_type:
            case "legacy":
                self.estimator = StateEstimator((0.0001, 0.007, 0.09, 0.005, 0.07))
                if (
                    settings.estimate_forces_motor
                    or settings.estimate_forces_dist
                    or settings.estimate_torques_dist
                ):
                    self.logger.warning(
                        "Legacy estimator does not support force or torque estimation!"
                    )
            case "ukf":
                self.input_needed = True
                self.estimator = KalmanFilter(
                    dt=1 / self.frequency,
                    model=settings.dynamics_model,  # mellinger_rpyt, fitted_DI_rpy
                    config=settings.drone_config,
                    estimate_forces_motor=settings.estimate_forces_motor,
                    estimate_forces_dist=settings.estimate_forces_dist,
                    estimate_torques_dist=settings.estimate_torques_dist,
                )
            case _:
                raise NotImplementedError(
                    f"Estimator type {self.settings.estimator_type} not implemented."
                )

        # Initialization
        self.logger.info("Waiting for initial measurement.")
        while not self._shutdown.is_set():
            with self._tf_msg_buffer.get_lock():
                data = np.asarray(self._tf_msg_buffer, dtype=np.float64, copy=True)
                self._tf_msg_buffer[0] = 0
            n_tf_msg, tf_timestamp, pos, quat = data[0], data[1], data[2:5], data[5:]

            if n_tf_msg >= 1:
                self.time_stamp_last_prediction = tf_timestamp
                self.time_stamp_last_correction = tf_timestamp
                self.estimator.set_state(pos, quat)
                self.logger.info("Initialized pos and quat.")
                break

        self.logger.info(f"Started estimator (process {os.getpid()})")

    def run(self):
        """Main estimator loop."""
        # TODO if first measurement => init estimator
        # self.estimator_init = False
        k = 0
        global_time = time.perf_counter()

        try:
            # self.logger.info(f"{self.estimator.data.pos=}")
            # Estimation loop
            while not self._shutdown.is_set():
                loop_start_time = time.time()

                with self._tf_msg_buffer.get_lock():
                    data = np.asarray(self._tf_msg_buffer, dtype=np.float64, copy=True)
                    self._tf_msg_buffer[0] = 0
                n_tf_msg, tf_timestamp, pos, quat = data[0], data[1], data[2:5], data[5:]
                with self._cmd_msg_buffer.get_lock():
                    data = np.asarray(self._cmd_msg_buffer, dtype=np.float64, copy=True)
                    self._cmd_msg_buffer[0] = 0
                n_cmd_messages, _, cmd = data[0], data[1], data[2:]

                if n_cmd_messages >= 1 and self.input_needed:
                    # The command is as it is sent to the drone, meaning for attitude interface:
                    # roll (deg), pitch (deg), yaw (deg), thrust (PWM)
                    # All the models run with rad and N, so we need to convert the RPYT command
                    cmd[..., -1] = cf2.pwm2force(cmd[..., -1], self.estimator.constants)
                    cmd[..., :-1] = np.deg2rad(cmd[..., :-1])
                    self.estimator.set_input(cmd)  # TODO # compare times?

                if n_tf_msg > 2:
                    self.logger.warning("Dropping tf messages because estimator loop can't keep up")
                    # TODO check for frequencies. If Vicon is running at higher frequency, of course estimator cant keep up

                if n_tf_msg >= 1:
                    dt = tf_timestamp - self.time_stamp_last_prediction
                    self.time_stamp_last_prediction = tf_timestamp

                    self.estimator.predict(dt)
                    self.estimator.correct(pos, quat)

                    # estimator_data = self.estimator.step(pos, quat, dt)

                time_stamp_now = time.time()
                dt = time_stamp_now - self.time_stamp_last_prediction
                self.time_stamp_last_prediction = time_stamp_now
                estimator_data = self.estimator.predict(dt)

                # Giving new estimate to publisher
                with self._pose_buffer.get_lock():
                    self._pose_buffer[:3] = estimator_data.pos
                    self._pose_buffer[3:] = estimator_data.quat
                with self._twist_buffer.get_lock():
                    self._twist_buffer[:3] = estimator_data.vel
                    self._twist_buffer[3:] = estimator_data.ang_vel
                if estimator_data.forces_motor is not None:
                    with self._forces_buffer.get_lock():
                        self._forces_buffer[:] = estimator_data.forces_motor
                if estimator_data.forces_dist is not None:
                    with self._wrench_buffer.get_lock():
                        self._wrench_buffer[:3] = estimator_data.forces_dist
                        if estimator_data.torques_dist is not None:
                            self._wrench_buffer[3:] = estimator_data.torques_dist
                self._publish_update.set()

                if self.settings.save_data:
                    append_state(self.data_est, time_stamp_now, estimator_data)
                    if n_tf_msg >= 1:
                        if n_cmd_messages >= 1 and self.input_needed:
                            append_measurement(self.data_meas, tf_timestamp, pos, quat, cmd)
                        if not self.input_needed:
                            append_measurement(self.data_meas, tf_timestamp, pos, quat, None)

                # if k % 100 == 0:
                #     self.logger.info(f"{estimator_data.forces_dist=}")

                remaining = (
                    (1 / self.frequency) - (time.time() - loop_start_time) - 1.25 * 1e-4
                )  # TODO remove magic number, replace with "controller"
                if k % 1000 == 999:
                    self.logger.info(f"Freq: {k / (time.perf_counter() - global_time)}")
                k += 1
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            self._shutdown.set()

    @staticmethod
    def _subscriber_loop(
        drone_name: str,
        _tf_msg_buffer: SynchronizedArray,
        _cmd_msg_buffer: SynchronizedArray,
        startup: Barrier,
        shutdown: Event,
    ):
        rclpy.init()
        node = rclpy.create_node("estimator_sub_" + drone_name)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        signal.signal(signal.SIGINT, lambda c, _: shutdown.set())  # Gracefully handle Ctrl-C

        def tf_callback(msg: TFMessage):
            tf = find_transform(msg.transforms, drone_name)
            if tf is None:
                node.get_logger().warning(
                    f"Drone {drone_name} could not have been found. Occluded?",
                    throttle_duration_sec=0.5,
                )
                return
            _, pos, quat = tf2array(tf)
            # We do not use the ros header time here because we need to establish an ordering
            # between observations and the model loop. We are not certain that the ros time stamp
            # is comparable to the system time. Therefore, we create a new timestamp using the os
            # time.
            # TODO: Subtract a constant time to account for [Vicon -> ros2 pub -> ros2 sub] delay.
            time_stamp = time.time()
            with _tf_msg_buffer.get_lock():
                _tf_msg_buffer[0] += 1
                _tf_msg_buffer[1] = time_stamp
                _tf_msg_buffer[2:5] = pos
                _tf_msg_buffer[5:9] = quat

        def cmd_callback(msg: Float64MultiArray):
            # The command is as it is sent to the drone, meaning for attitude interface:
            # roll (deg), pitch (deg), yaw (deg), thrust (PWM)
            with _cmd_msg_buffer.get_lock():
                _cmd_msg_buffer[0] += 1
                _cmd_msg_buffer[1] = time.time()
                _cmd_msg_buffer[2:] = msg.data

        sub_tf = node.create_subscription(TFMessage, "/tf", tf_callback, qos_profile=qos_profile)
        sub_cmd = node.create_subscription(
            Float64MultiArray,
            f"/drones/{drone_name}/command",
            cmd_callback,
            qos_profile=qos_profile,
        )
        startup.wait(10.0)  # Register this process as ready for startup barrier

        while not shutdown.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
        sub_tf.destroy()
        sub_cmd.destroy()
        node.destroy_node()

    @staticmethod
    def _publisher_loop(
        drone_name: str,
        pose_buffer: SynchronizedArray,
        twist_buffer: SynchronizedArray,
        forces_buffer: SynchronizedArray,
        wrench_buffer: SynchronizedArray,
        update: Event,
        startup: Barrier,
        shutdown: Event,
    ):
        rclpy.init()
        node = rclpy.create_node("estimator_sub_" + drone_name)
        # TODO check if pubs are actually needed?
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # pos, quat
        pub_pose = node.create_publisher(
            PoseStamped, f"/drones/{drone_name}/estimate/pose", qos_profile=qos_profile
        )
        # vel, ang_vel
        pub_twist = node.create_publisher(
            TwistStamped, f"/drones/{drone_name}/estimate/twist", qos_profile=qos_profile
        )
        # f_motors
        pub_forces = node.create_publisher(
            Float64MultiArray, f"/drones/{drone_name}/estimate/forces", qos_profile=qos_profile
        )
        # f_dis, t_dis
        pub_wrench = node.create_publisher(
            WrenchStamped, f"/drones/{drone_name}/estimate/wrench", qos_profile=qos_profile
        )
        # marker (only for rviz)
        pub_markers = node.create_publisher(
            MarkerArray, f"/drones/{drone_name}/estimate/marker_array", qos_profile=qos_profile
        )
        signal.signal(signal.SIGINT, lambda c, _: shutdown.set())  # Gracefully handle Ctrl-C
        startup.wait(10.0)  # Register this process as ready for startup barrier

        # TODO clear arrays???
        while not shutdown.is_set():
            if not update.wait(0.5):
                continue
            time_stamp = time.time()
            update.clear()

            pose = np.asarray(pose_buffer, dtype=np.float64, copy=True)
            pose_stamped = create_pose(time_stamp, drone_name, pose[:3], pose[3:])
            pub_pose.publish(pose_stamped)

            twist = np.asarray(twist_buffer, dtype=np.float64, copy=True)
            twist_stamped = create_twist(time_stamp, drone_name, twist[:3], twist[3:])
            pub_twist.publish(twist_stamped)

            forces = np.asarray(forces_buffer, dtype=np.float64, copy=True)
            # This type doesn't have a stamp!
            forces_array = create_array(time_stamp, drone_name, forces)
            pub_forces.publish(forces_array)

            wrench = np.asarray(wrench_buffer, dtype=np.float64, copy=True)
            wrench_stamped = create_wrench(time_stamp, drone_name, wrench[:3], wrench[3:])
            pub_wrench.publish(wrench_stamped)

            markers = create_marker_array(
                time_stamp,
                drone_name,
                pose[:3],
                pose[3:],
                twist[:3],
                twist[3:],
                wrench[:3],
                wrench[3:],
            )
            pub_markers.publish(markers)

        pub_pose.destroy()
        pub_twist.destroy()
        pub_forces.destroy()
        pub_wrench.destroy()
        node.destroy_node()

    def close(self):
        """TODO."""
        self.logger.info(f"Estimator {self.settings.drone_name} shutdown")
        self._shutdown.set()
        self._sub_process.join()
        self._pub_process.join()

        if self.settings.save_data:
            self.logger.info("Saving data...")
            filename = f"data_{self.settings.drone_name}_"
            info = f"{self.settings.estimator_type}"
            if self.settings.estimator_type != "legacy":
                info = info + f"_{self.settings.dynamics_model}"
            with open(filename + info + ".pkl", "wb") as f:
                pickle.dump(self.data_est, f)
            with open(filename + "measurement" + ".pkl", "wb") as f:
                pickle.dump(self.data_meas, f)


def launch_estimators(estimators: dict):
    """TODO."""
    processes = []
    seen_drone_names = []
    ctx = mp.get_context("spawn")
    shutdown = ctx.Event()

    try:
        for k, settings in estimators.items():
            name = settings.drone_name
            if name in seen_drone_names:
                print(f"[ESTIMATOR_{name}]  Estimator for {name} already existing. Check settings")
                continue
            seen_drone_names.append(name)
            # not sure if daemon should be True or False
            p = ctx.Process(target=launch_node, args=(settings, shutdown))
            processes.append(p)
            p.start()

        try:
            while True:
                time.sleep(10.0)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Terminating nodes...")

    finally:
        shutdown.set()

        for p in processes:
            p.join(timeout=2)

        for p in processes:
            if p.is_alive():
                print(f"Force terminating process {p.pid}")
                p.terminate()
                p.join()

        if rclpy.ok():
            rclpy.shutdown()
        print("All nodes terminated.")


def launch_node(settings: Munch, stop_event: Event):
    """TODO."""
    rclpy.init()
    estimator = MPEstimator(settings)
    try:
        estimator.run()
    finally:
        estimator.close()


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=3)  # TODO remove
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settings", default="ros_nodes/estimators.toml", help="Path to Settings file"
    )
    args = parser.parse_args()

    path = Path(__file__).parents[1] / args.settings
    with open(path, "r") as f:
        estimators = munchify(toml.load(f))

    # Add debug to each estimator (if not already in place)
    for key, val in estimators.items():
        if not key.startswith("estimator"):
            continue
        for global_key, global_val in estimators.get("global", {}).items():
            if global_key not in val:
                val[global_key] = global_val

    estimators = munchify({k: v for k, v in estimators.items() if k.startswith("estimator")})

    launch_estimators(estimators)
