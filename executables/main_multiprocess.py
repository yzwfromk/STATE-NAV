#!/usr/bin/env python3
"""
Parent Process - Spawns WorldModel and Planner as separate processes

Architecture:
- Parent process spawns WorldModel and Planner as separate processes
- Each child runs its own ROS node
- Children share memory via named shared memory (statenav_meta_shm, etc.)
- True parallel execution (no context switching overhead)

Usage:
    ros2 run statenav_global multiprocess
"""

import multiprocessing as mp
import signal
import sys

# Import from main_global_planning
# Note: Import happens inside worker functions to avoid ROS initialization issues


def world_model_worker():
    """Worker function for WorldModel process (writer)"""
    import rclpy
    from executables.main_worldmodel import WorldModelNode

    # Initialize ROS2 in THIS process
    rclpy.init()
    world_model_node = None  # Initialize to handle creation failures

    try:
        # Create WorldModelNode with shared metadata
        # Note: ROS2 nodes always initialize via __init__, no init_ros parameter
        world_model_node = WorldModelNode(use_sim_time=True)
        world_model_node.get_logger().info(f"[WorldModel] WorldModel process started (PID: {mp.current_process().pid})")

        # Run world model node
        world_model_node.run()
    finally:
        if world_model_node is not None:
            world_model_node.get_logger().info("[WorldModel] WorldModel process shutting down")
        if rclpy.ok():
            rclpy.shutdown()


def planner_worker():
    """Worker function for Planner process (reader)"""
    import rclpy
    from executables.main_global_planning import PlanningNode

    # Initialize ROS2 in THIS process
    rclpy.init()
    planning_node = None  # Initialize to handle creation failures

    try:
        planning_node = PlanningNode(use_sim_time=True)
        planning_node.get_logger().info(f"[Planner] Planner process started (PID: {mp.current_process().pid})")

        # Run planner node
        planning_node.run()
    finally:
        if planning_node is not None:
            planning_node.get_logger().info("[Planner] Planner process shutting down")
        if rclpy.ok():
            rclpy.shutdown()


def signal_handler(signum, frame):
    """Handle signals to gracefully shutdown children"""
    print(f"\nReceived signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """
    Parent process: Creates shared resources and spawns children
    """
    # Set multiprocessing start method (important for ROS compatibility)
    # 'spawn' works on all platforms, 'fork' is faster on Unix but can cause issues with ROS
    mp.set_start_method('spawn', force=True)
    
    print("=" * 80)
    print("Parent Process: Creating shared resources and spawning children")
    print("=" * 80)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Spawn child processes
    print("Spawning WorldModel process (writer)...")
    world_model_process = mp.Process(
        target=world_model_worker,
        name='WorldModel'
    )

    print("Spawning Planner process (reader)...")
    planner_process = mp.Process(
        target=planner_worker,
        name='Planner'
    )
    
    # Start processes (they run in parallel!)
    world_model_process.start()
    planner_process.start()
    
    print(f"WorldModel PID: {world_model_process.pid}")
    print(f"Planner PID: {planner_process.pid}")
    print("\nChildren running in parallel. Press Ctrl+C to stop.\n")
    
    try:
        # Wait for children to finish (or until interrupted)
        world_model_process.join()
        planner_process.join()
    except KeyboardInterrupt:
        print("\nInterrupted by user, terminating children...")
        world_model_process.terminate()
        planner_process.terminate()
        world_model_process.join(timeout=5)
        planner_process.join(timeout=5)
    
    print("Parent process exiting")


if __name__ == '__main__':
    main()
