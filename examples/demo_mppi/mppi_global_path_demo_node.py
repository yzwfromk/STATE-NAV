#!/usr/bin/env python3
"""Isolated ROS 2 MPPI visualization driven by the live global path and map.

The node reads the production ``/global_path`` and shared traversability map,
but advances its own first-order simulated robot.  Every published output is
under ``/mppi_demo``; it never publishes to the robot command topic.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path as FilePath
import sys
from typing import Any, Optional

import numpy as np

DEMO_DIRECTORY = FilePath(__file__).resolve().parent
TRAVERSABILITY_DIRECTORY = DEMO_DIRECTORY.parent
PROJECT_ROOT = TRAVERSABILITY_DIRECTORY.parent
sys.path.insert(0, str(TRAVERSABILITY_DIRECTORY))
sys.path.insert(0, str(PROJECT_ROOT))
WORKSPACE_SOURCE_DIRECTORY = TRAVERSABILITY_DIRECTORY.parent.parent
_STATE_NAV_SOURCE = WORKSPACE_SOURCE_DIRECTORY / "state_nav"
if _STATE_NAV_SOURCE.is_dir():
    sys.path.insert(0, str(_STATE_NAV_SOURCE))

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from omegaconf import OmegaConf
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from demo_mppi.demo_map_provider import create_demo_map  # noqa: E402
from demo_mppi.demo_mppi_tracking import (  # noqa: E402
    ClosedLoopMPPITracker,
    ClosedLoopTrackingConfig,
)
from demo_mppi.demo_result_plot import save_demo_result_plot  # noqa: E402
from statenav_mppi.mppi_core.mppi_dynamics import lag_response_factor
from statenav_mppi.mppi_utils.mppi_path_geometry import sanitize_path
from statenav_mppi.mppi_planners.mppi_planner import MPPIConfig, MPPIPlanner


def path_points_from_message(message: Path) -> np.ndarray:
    """Return a validated ``(N, 2)`` path from a ROS ``nav_msgs/Path``."""
    points = np.asarray([
        [pose.pose.position.x, pose.pose.position.y]
        for pose in message.poses
    ], dtype=np.float32).reshape(-1, 2)
    return sanitize_path(points)


def _default_config_path() -> FilePath:
    workspace_config = (
        _STATE_NAV_SOURCE / "statenav_global/configs/planning_config.yaml"
    )
    if workspace_config.is_file():
        return workspace_config
    import statenav_global

    return FilePath(statenav_global.__file__).resolve().parent / "configs/planning_config.yaml"


def demo_mppi_config(
    source: Any,
    *,
    device: str,
) -> MPPIConfig:
    """Apply demo-only overrides without coupling MPPI rollout to loop time."""
    return replace(
        MPPIConfig.from_source(source),
        device=str(device),
    )


def demo_plant_response(control_period: float, lag_time: float) -> float:
    """Return the production first-order response over one demo control tick."""
    return lag_response_factor(control_period, lag_time)


class MPPIGlobalPathDemoNode(Node):
    """Visualize production MPPI on an isolated simulated robot."""

    def __init__(self) -> None:
        super().__init__("mppi_global_path_demo")

        self.declare_parameter("planning_config_path", str(_default_config_path()))
        config_path = FilePath(str(
            self.get_parameter("planning_config_path").value)).expanduser().resolve()
        cfg = OmegaConf.load(config_path)

        self.declare_parameter("path_topic", "/global_path")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("use_shared_map", True)
        self.declare_parameter("reset_on_new_path", False)
        self.declare_parameter("control_period", float(cfg.ctrl_period))
        control_period = float(self.get_parameter("control_period").value)
        self.declare_parameter(
            "plant_response",
            demo_plant_response(control_period, float(cfg.mppi_lag_time)),
        )
        self.declare_parameter("goal_tolerance_m", 0.32)
        self.declare_parameter("max_v", float(cfg.maximum_cmd_v))
        self.declare_parameter("max_w", float(cfg.maximum_cmd_w))
        self.declare_parameter("device", str(getattr(cfg, "mppi_device", "auto")))
        self.declare_parameter("flat_map_half_extent_m", 20.0)
        self.declare_parameter("flat_map_resolution_m", 0.25)
        self.declare_parameter("flat_map_heading_layers", 36)

        self._path_topic = str(self.get_parameter("path_topic").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._reset_on_new_path = bool(
            self.get_parameter("reset_on_new_path").value)
        self._elapsed_s = 0.0
        self._last_source: Optional[str] = None
        self._tracker: Optional[ClosedLoopMPPITracker] = None
        self._goal_result_saved = False
        self._goal_result_path = (
            TRAVERSABILITY_DIRECTORY.parent / "results" / "mppi_demo_goal_reached.png"
        )

        # Register before map construction. Shared-map attachment may wait for
        # the writer, so DDS retains the newest path published during that wait.
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Path, self._path_topic, self._path_callback, path_qos)
        self.create_subscription(Bool, "/mppi_demo/reset", self._reset_callback, 10)

        mppi_config = demo_mppi_config(
            cfg,
            device=str(self.get_parameter("device").value),
        )
        tracking_config = ClosedLoopTrackingConfig(
            control_period=control_period,
            max_v=float(self.get_parameter("max_v").value),
            max_w=float(self.get_parameter("max_w").value),
            plant_response=float(self.get_parameter("plant_response").value),
            goal_tolerance_m=float(
                self.get_parameter("goal_tolerance_m").value),
            trail_length=int(getattr(cfg, "mppi_trajectory_trail_length", 500)),
        )

        use_shared_map = bool(
            self.get_parameter("use_shared_map").value)
        map_description = (
            "shared traversability map" if use_shared_map
            else "standalone flat map")
        self.get_logger().info(
            f"Connecting isolated demo to the {map_description}...")
        self._backend, self._map = create_demo_map(
            use_shared_map=use_shared_map,
            half_extent_m=float(
                self.get_parameter("flat_map_half_extent_m").value),
            resolution_m=float(
                self.get_parameter("flat_map_resolution_m").value),
            heading_layers=int(
                self.get_parameter("flat_map_heading_layers").value),
            max_v=float(self.get_parameter("max_v").value),
            max_w=float(self.get_parameter("max_w").value),
            lattice_resolution_m=float(mppi_config.travgraph_resolution),
        )
        self._planner = MPPIPlanner(mppi_config, self._map)
        self._tracker = ClosedLoopMPPITracker(self._planner, tracking_config)

        self._pose_pub = self.create_publisher(
            PoseStamped, "/mppi_demo/pose", 10)
        self._odom_pub = self.create_publisher(
            Odometry, "/mppi_demo/odom", 10)
        self._cmd_pub = self.create_publisher(
            TwistStamped, "/mppi_demo/cmd_vel", 10)
        self._trail_pub = self.create_publisher(
            Path, "/mppi_demo/trail", 10)
        self._target_pub = self.create_publisher(
            PoseStamped, "/mppi_demo/local_target", 10)
        self._metrics_pub = self.create_publisher(
            String, "/mppi_demo/metrics", 10)

        self.create_timer(control_period, self._control_tick)
        self.get_logger().info(
            f"Waiting for the next {self._path_topic} publication; if a volatile path "
            "was published before this demo started, trigger a replan. "
            "Outputs are isolated under /mppi_demo/*")

    def _path_callback(self, message: Path) -> None:
        if self._tracker is None:
            self.get_logger().warning(
                "Ignoring global path while shared-map initialization is incomplete")
            return
        try:
            points = path_points_from_message(message)
            changed = self._tracker.set_path(
                points,
                reset_pose=(
                    self._reset_on_new_path and self._tracker.initialized),
            )
        except ValueError as exc:
            self.get_logger().warning(f"Rejected global path: {exc}")
            return
        if changed:
            self._frame_id = message.header.frame_id or self._frame_id
            self._elapsed_s = 0.0
            self._goal_result_saved = False
            self.get_logger().info(
                f"Accepted global path: {points.shape[0]} points, "
                f"frame={self._frame_id}")

    def _reset_callback(self, message: Bool) -> None:
        if (
            message.data and self._tracker is not None
            and self._tracker.initialized
        ):
            self._tracker.reset()
            self._planner.reset_visualization()
            self._elapsed_s = 0.0
            self._goal_result_saved = False
            self.get_logger().info(
                "Reset isolated MPPI robot to the global-path start")

    def _save_goal_result_once(self, step: Any) -> None:
        """Persist the complete demo summary on the first goal-reaching tick."""
        if self._goal_result_saved or not bool(step.goal_reached):
            return
        if self._tracker is None:
            return
        try:
            # step.pose is (x, y, yaw); the background is the layer for the goal-time yaw.
            terrain_snapshot = self._planner.visualization_snapshot(
                float(step.pose[2]))
            output_path = save_demo_result_plot(
                output_path=self._goal_result_path,
                path=self._tracker.path,
                trail=self._tracker.trail,
                commands=self._tracker.commands,
                errors=self._tracker.errors,
                solve_ms=self._tracker.solve_ms,
                control_period=self._tracker.config.control_period,
                goal_reached=True,
                terrain_snapshot=terrain_snapshot,
            )
        except Exception as exc:
            self.get_logger().warning(
                f"[mppi_demo] could not save goal result plot "
                f"({type(exc).__name__}: {exc}); will retry next tick",
                throttle_duration_sec=1.0,
            )
            return
        self._goal_result_saved = True
        self.get_logger().info(f"Saved goal result plot to {output_path}")

    @staticmethod
    def _set_pose(message: PoseStamped, pose: np.ndarray) -> None:
        message.pose.position.x = float(pose[0])
        message.pose.position.y = float(pose[1])
        message.pose.orientation.z = math.sin(float(pose[2]) * 0.5)
        message.pose.orientation.w = math.cos(float(pose[2]) * 0.5)

    def _path_message(self, points: np.ndarray, stamp) -> Path:
        message = Path()
        message.header.stamp = stamp
        message.header.frame_id = self._frame_id
        for point in np.asarray(points):
            pose = PoseStamped()
            pose.header = message.header
            yaw = float(point[2]) if point.shape[0] > 2 else 0.0
            self._set_pose(pose, np.array([
                float(point[0]), float(point[1]), yaw]))
            message.poses.append(pose)
        return message

    def _publish(self, step) -> None:
        assert self._tracker is not None
        stamp = self.get_clock().now().to_msg()
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self._frame_id
        self._set_pose(pose, step.pose)
        self._pose_pub.publish(pose)

        odom = Odometry()
        odom.header = pose.header
        odom.child_frame_id = "mppi_demo/base_link"
        odom.pose.pose = pose.pose
        odom.twist.twist.linear.x = float(step.velocity[0])
        odom.twist.twist.angular.z = float(step.velocity[1])
        self._odom_pub.publish(odom)

        command = TwistStamped()
        command.header = pose.header
        command.twist.linear.x = float(step.command[0])
        command.twist.angular.z = float(step.command[1])
        self._cmd_pub.publish(command)
        self._trail_pub.publish(self._path_message(self._tracker.trail, stamp))

        target = PoseStamped()
        target.header = pose.header
        self._set_pose(target, np.array([
            float(step.target[0]), float(step.target[1]), 0.0]))
        self._target_pub.publish(target)

        metrics = self._tracker.metrics()
        metrics.update({"source": step.source})
        metrics_message = String()
        metrics_message.data = json.dumps(metrics, sort_keys=True)
        self._metrics_pub.publish(metrics_message)

    def _control_tick(self) -> None:
        if self._tracker is None or not self._tracker.initialized:
            return
        try:
            step = self._tracker.advance()
        except Exception as exc:
            warning = (
                f"[mppi_demo] {type(exc).__name__}: {exc} "
                "— holding isolated robot and retrying next tick")
            step = self._tracker.hold("hold(demo_error)", (warning,))
        try:
            self._elapsed_s += self._tracker.config.control_period
            self._publish(step)
            self._save_goal_result_once(step)
            for warning in step.warnings:
                self.get_logger().warning(
                    warning, throttle_duration_sec=1.0)
            if step.diagnostic is not None:
                self.get_logger().info(
                    step.diagnostic,
                    throttle_duration_sec=(
                        self._tracker.config.control_period * 0.9),
                )
            self._planner.publish_plan(
                self._tracker.path,
                step.target,
                step.pose,
                float(step.command[0]),
                float(step.command[1]),
                mode="wp_tracking",
                time=self._elapsed_s,
            )
            if step.source != self._last_source:
                self.get_logger().info(
                    f"source={step.source} progress={step.path_progress:.1%} "
                    f"cross_track={step.cross_track_m:.3f}m "
                    f"goal_distance={step.goal_distance_m:.3f}m")
                self._last_source = step.source
        except Exception as exc:
            self._tracker.velocity[:] = 0.0
            self.get_logger().error(
                f"[mppi_demo] output/visualization error "
                f"({type(exc).__name__}: {exc}); retrying next tick",
                throttle_duration_sec=1.0,
            )

    def destroy_node(self) -> bool:
        self._backend.cleanup()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node: Optional[MPPIGlobalPathDemoNode] = None
    try:
        node = MPPIGlobalPathDemoNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
