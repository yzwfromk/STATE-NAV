#!/usr/bin/env python3
import threading
import time
import traceback

import rclpy
import rclpy.parameter
from rclpy.node import Node
from rclpy.clock import ClockType
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Bool, String

from omegaconf import OmegaConf
from pathlib import Path as PathLib
import statenav_global
# Shared QoS ladder — single source of truth for the wire contract across statenav / gsa.
from statenav_fsm.qos_profiles import (
    qos_profile_level2,
    qos_profile_level3,
)
from statenav_global.core.config import load_planning_config
cfg = load_planning_config()   # $STATENAV_PLANNING_CONFIG, else the packaged planning_config.yaml

from statenav_fsm.NavigationFSM import NavigationFSM, FSMState


class FSMNode(Node):
    """
    Standalone ROS2 node that owns the NavigationFSM.

    Subscribes to:
        /robot/pose          (PoseStamped)       — robot position
        /target_obstacle     (Float32MultiArray)  — [n, x0, y0, r0] from global planner
        /interaction_failure (Bool)              — failure vote from IsaacLab's task_failure (robot stability)
        /interaction_failure_progress (Bool)     — failure vote from the global planner's progress monitor
        /interaction_success (Bool)              — success vote from local planner

    Publishes:
        /fsm_state  (String) — "INITIALIZATION_SETTLING", "INITIALIZATION_ROTATE",
                               "INITIALIZATION_HOLD", "WP_TRACKING", "INTERACTION",
                               "FAILURE_RECOVERY", or "WP_TRACKING_SUCCESS"
                               (WP_TRACKING_SUCCESS is published for _TRANSITION_TAG_DURATION
                               seconds after a successful INTERACTION exit so subscribers don't miss it)
                               The INITIALIZATION_* strings carry the init sub-phase: the local
                               planner issues a different command in each, while consumers that
                               only need the base state collapse on the "INITIALIZATION" prefix.
    """

    def __init__(self, use_sim_time: bool = False):
        super().__init__(
            'fsm_node',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, use_sim_time)
            ]
        )

        self.fsm = NavigationFSM()
        # Where the FSM's timing values actually came from. NavigationFSM falls back to built-in
        # defaults when no planning_config.yaml is found, which silently changes the threshold
        # below (interaction_time_limit 15 -> 10 s), so the source and the resolved value are
        # stated once at startup. The two *_flag_window_duration keys are gone: every
        # interaction vote is now a ONE-SHOT latch (the first True decides), so there is no
        # window left to report.
        self.get_logger().info(
            f"[FSM] config source: {self.fsm.config_source} | "
            f"interaction_time_limit={self.fsm._interaction_time_limit}s | "
            f"interaction votes are ONE-SHOT (first True on /interaction_success, "
            f"/interaction_failure or /interaction_failure_progress decides the episode)")
        # The init schedule is absolute sim time, so printing the resolved boundaries once is what
        # makes a run's /fsm_state timeline checkable against the config that produced it.
        self.get_logger().info(
            f"[FSM] init schedule (sim time, from /clock): "
            f"SETTLING [0, {self.fsm._cloud_settle_time:.2f}) -> "
            f"ROTATE [{self.fsm._cloud_settle_time:.2f}, {self.fsm._init_rotate_end:.2f}) -> "
            f"HOLD [{self.fsm._init_rotate_end:.2f}, {self.fsm._initialization_time:.2f}) -> "
            f"WP_TRACKING. Timing only: the local planner issues every /robot/cmd_vel of these "
            f"phases.")

        # QoS from the shared ladder (statenav_fsm.qos_profiles):
        #   level3 -> /robot/pose, the continuous pose stream the FSM's distance-based
        #             transitions depend on (RELIABLE, 10-deep so a burst is not dropped)
        #   level2 -> the low-rate state / one-shot event flags below, where only the newest
        #             value is meaningful (RELIABLE, latest-only)
        self.create_subscription(PoseStamped, '/robot/pose',
                                 self._pose_callback, qos_profile_level3)
        self.create_subscription(Float32MultiArray, '/target_obstacle',
                                 self._target_obstacle_callback, qos_profile_level2)
        self.create_subscription(Bool, '/interaction_failure',
                                 self.fsm.interaction_failure_callback, qos_profile_level2)
        # Second failure source on its OWN topic: IsaacLab streams task_failure on
        # /interaction_failure at step rate (almost always False), which would reset a shared
        # streak on every message. The two votes are OR-ed inside NavigationFSM instead.
        self.create_subscription(Bool, '/interaction_failure_progress',
                                 self.fsm.interaction_progress_failure_callback, qos_profile_level2)
        self.create_subscription(Bool, '/interaction_success',
                                 self.fsm.interaction_success_callback, qos_profile_level2)
        self.create_subscription(Bool, '/failure_recovered',
                                 self.fsm.failure_recovered_callback, qos_profile_level2)

        self.fsm_state_pub = self.create_publisher(String, '/fsm_state', qos_profile_level2)
        self._prev_fsm_state = self.fsm.FSM_STATE
        self._transition_tag = None          # "WP_TRACKING_SUCCESS" (only success uses a transition tag)
        self._transition_tag_until = 0.0     # ROS time (seconds) until which tag is published
        self._TRANSITION_TAG_DURATION = cfg.fsm_transition_tag_duration  # seconds

        clock_type = self.get_clock().clock_type
        if clock_type != ClockType.ROS_TIME:
            self.get_logger().warning(
                f"[FSM] use_sim_time is OFF (clock={clock_type.name}). "
                "Transition tag duration uses wall time. "
                "Pass --ros-args -p use_sim_time:=true to use /clock."
            )
        else:
            self.get_logger().info("[FSM] Using ROS time from /clock")

        self.get_logger().info("[FSM] FSMNode initialized")

    def _pose_callback(self, msg):
        self.fsm.robot_x = msg.pose.position.x
        self.fsm.robot_y = msg.pose.position.y

    def _target_obstacle_callback(self, msg):
        """Parse [n, x0, y0, r0] and store the target obstacle."""
        data = list(msg.data)
        if len(data) < 1 or int(data[0]) == 0 or len(data) < 4:
            self.fsm.target_obstacle_x = None
            self.fsm.target_obstacle_y = None
            self.fsm.target_obstacle_radius = None
            return
        self.fsm.target_obstacle_x = data[1]
        self.fsm.target_obstacle_y = data[2]
        self.fsm.target_obstacle_radius = data[3]

    def run(self):
        self.get_logger().info("[FSM] FSMNode running")

        # Spun through a wrapper rather than raw rclpy.spin: an exception escaping ANY
        # subscription callback tears down this daemon thread, after which the node keeps
        # publishing /fsm_state from the loop below while processing no topic at all — a DEAF
        # FSM whose only surviving transition is the INTERACTION timeout, which looks exactly
        # like "the failure vote is ignored". Report it instead of failing silently.
        def _spin_and_report():
            try:
                rclpy.spin(self)
            except Exception:
                self.get_logger().error(
                    "[FSM] spin thread DIED — no further topic callback will be processed "
                    "(/robot/pose, /target_obstacle, /interaction_failure, /interaction_success "
                    "and /failure_recovered are all ignored from here on):\n"
                    + traceback.format_exc())

        spin_thread = threading.Thread(target=_spin_and_report, daemon=True)
        spin_thread.start()
        spin_alive = True   # one-shot latch so the deafness below is reported exactly once

        while rclpy.ok():
            current_time = self.get_clock().now().nanoseconds / 1e9
            # Deafness guard: name the failure at the moment it happens. Covers a spin thread
            # that ended WITHOUT an exception as well (executor shutdown), which the wrapper
            # above cannot see.
            if spin_alive and not spin_thread.is_alive():
                spin_alive = False
                self.get_logger().error(
                    "[FSM] the spin thread is no longer running: this node is DEAF. Robot pose, "
                    "target obstacle and every interaction vote are now ignored, so INTERACTION "
                    "can only end on interaction_time_limit.")
            self.fsm.update(current_time)

            # Failure now transitions INTERACTION -> FAILURE_RECOVERY (proper FSM state).
            # Only success reaches WP_TRACKING directly from INTERACTION.
            if (self._prev_fsm_state == FSMState.INTERACTION
                    and self.fsm.FSM_STATE == FSMState.WP_TRACKING
                    and self.fsm.last_exit_success is not None):
                self._transition_tag = "WP_TRACKING_SUCCESS"
                self._transition_tag_until = current_time + self._TRANSITION_TAG_DURATION
                self.fsm.last_exit_success = None

            state_msg = String()
            if current_time < self._transition_tag_until:
                state_msg.data = self._transition_tag
            else:
                # published_state_name(), not FSM_STATE.name: INITIALIZATION publishes its
                # sub-phase so the local planner knows which init command to issue.
                state_msg.data = self.fsm.published_state_name()
            self._prev_fsm_state = self.fsm.FSM_STATE
            self.fsm_state_pub.publish(state_msg)

            # Loop rate limit. This loop had NO throttle: it ran as fast as the interpreter
            # allowed, pegging a core and publishing /fsm_state at that same unbounded rate.
            # WALL-clock sleep on purpose — self.get_clock() is sim time, which can stop or run
            # at any factor, and a sim-time sleep here would freeze the deafness guard with it.
            # 50 Hz is well above every consumer's need (/fsm_state subscribers are level2,
            # depth 1, and only act on state CHANGES) and above the FSM's own timing resolution:
            # interaction_time_limit is checked once per iteration, so the timeout is now
            # detected within 20 ms of expiry instead of instantly — immaterial at a 15 s limit.
            time.sleep(0.02)


def main():
    rclpy.init()
    node = None
    try:
        node = FSMNode(use_sim_time=True)
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.get_logger().info("[FSM] FSMNode shutting down")
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
