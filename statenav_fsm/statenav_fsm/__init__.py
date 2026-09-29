"""Portable STATE-NAV pieces shared across repos (statenav, gsa, elevation_mapping,
IsaacLab, Bipedal_Planning_MPC).

This __init__ is INTENTIONALLY EMPTY — it must not import its submodules.

Importing a submodule always executes this file first, so any re-export here would
force every consumer to satisfy every submodule's dependencies.  Concretely: a node
that only wants the QoS ladder (`from statenav_fsm.qos_profiles import ...`) would be
forced to have omegaconf and a readable planning config just to create a publisher.
That is exactly the failure mode statenav_global.__init__ has, where importing any
submodule pulls torch and every ROS message package.

Import submodules explicitly instead:
    from statenav_fsm.qos_profiles import qos_profile_level1, qos_profile_level2
    from statenav_fsm.NavigationFSM import NavigationFSM, FSMState
"""
