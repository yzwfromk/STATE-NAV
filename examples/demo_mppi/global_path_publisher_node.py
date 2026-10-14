#!/usr/bin/env python3
"""Publish configurable synthetic global paths for the isolated MPPI demo."""

from __future__ import annotations

import math
from pathlib import Path as FilePath
import sys
from typing import Any, Optional

import numpy as np


DEMO_DIRECTORY = FilePath(__file__).resolve().parent
TRAVERSABILITY_DIRECTORY = DEMO_DIRECTORY.parent
sys.path.insert(0, str(TRAVERSABILITY_DIRECTORY))

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from demo_mppi.global_path_scenarios import (  # noqa: E402
    generate_path_scenario,
    path_tangent_headings,
)


def build_path_message(
    points: np.ndarray,
    headings: np.ndarray,
    frame_id: str,
    stamp: Any,
) -> Path:
    """Convert finite world xy and tangent headings to a stamped ROS path."""
    values = np.asarray(points, dtype=np.float64)
    yaw = np.asarray(headings, dtype=np.float64)
    frame = str(frame_id).strip()
    if not frame:
        raise ValueError("frame_id must not be empty")
    if values.ndim != 2 or values.shape[1:] != (2,) or values.shape[0] < 2:
        raise ValueError("points must have shape (N, 2) with N >= 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("path points must be finite")
    if yaw.shape != (values.shape[0],) or not np.all(np.isfinite(yaw)):
        raise ValueError("heading count must match finite path points")

    message = Path()
    message.header.frame_id = frame
    message.header.stamp = stamp
    for point, heading in zip(values, yaw):
        pose = PoseStamped()
        pose.header.frame_id = frame
        pose.header.stamp = stamp
        pose.pose.position.x = float(point[0])
        pose.pose.position.y = float(point[1])
        pose.pose.orientation.z = math.sin(float(heading) * 0.5)
        pose.pose.orientation.w = math.cos(float(heading) * 0.5)
        message.poses.append(pose)
    return message


class SyntheticGlobalPathPublisher(Node):
    """Periodically publish one immutable synthetic path scenario."""

    def __init__(self) -> None:
        super().__init__("synthetic_global_path_publisher")
        self.declare_parameter("path_type", "straight")
        self.declare_parameter("spacing", 0.25)
        self.declare_parameter("scale", 1.0)
        self.declare_parameter("origin_x", 0.0)
        self.declare_parameter("origin_y", 0.0)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("path_topic", "/global_path")
        self.declare_parameter("publish_rate_hz", 1.0)

        self._path_type = str(self.get_parameter("path_type").value).strip()
        self._frame_id = str(self.get_parameter("frame_id").value).strip()
        self._path_topic = str(
            self.get_parameter("path_topic").value).strip()
        publish_rate_hz = float(
            self.get_parameter("publish_rate_hz").value)
        if not self._frame_id:
            raise ValueError("frame_id must not be empty")
        if not self._path_topic:
            raise ValueError("path_topic must not be empty")
        if not math.isfinite(publish_rate_hz) or publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be finite and positive")

        self._points = generate_path_scenario(
            self._path_type,
            spacing=float(self.get_parameter("spacing").value),
            scale=float(self.get_parameter("scale").value),
            origin_x=float(self.get_parameter("origin_x").value),
            origin_y=float(self.get_parameter("origin_y").value),
        )
        self._headings = path_tangent_headings(self._points)
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._publisher = self.create_publisher(
            Path, self._path_topic, path_qos)
        self._publish()
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._publish)

        path_length = float(np.sum(np.linalg.norm(
            np.diff(self._points, axis=0), axis=1)))
        self.get_logger().info(
            f"Publishing path_type={self._path_type} "
            f"points={self._points.shape[0]} length={path_length:.2f}m "
            f"topic={self._path_topic} frame={self._frame_id} "
            f"rate={publish_rate_hz:.2f}Hz")

    def _publish(self) -> None:
        message = build_path_message(
            self._points,
            self._headings,
            self._frame_id,
            self.get_clock().now().to_msg(),
        )
        self._publisher.publish(message)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[SyntheticGlobalPathPublisher] = None
    try:
        node = SyntheticGlobalPathPublisher()
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
