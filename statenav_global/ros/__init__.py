"""ROS 2 adapters: the only statenav_global code that may import rclpy / ROS message packages.

Everything else in statenav_global must import without a ROS environment (checked by
tests/check_package_isolation.py). This __init__ is INTENTIONALLY EMPTY; import submodules
explicitly, e.g.:
    from statenav_global.ros.ros_utils import Rviz_vis_graph
"""
