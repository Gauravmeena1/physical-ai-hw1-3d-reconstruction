"""Config-driven Habitat-Sim core engine shared across course homeworks.

Public surface (the Phase-0 interface contract — see HW1 specification). `viewer` is NOT
re-exported here: it is the only module that imports pygame and must be imported
explicitly (`from simulator import viewer`) so headless callers never touch it.
"""

from simulator.config import load_config
from simulator.effects import (
    OUTSIDE,
    ZoneScheduler,
    apply_depth_faults,
    apply_depth_sensor,
    apply_lighting,
    depth_to_vis,
    light_exposure,
    process_observations,
    semantic_to_vis,
    zone_frame_counts,
    zone_frame_labels,
)
from simulator.engine import Engine, add_start_marker, make_cfg, make_sensor_spec
from simulator.replay import (
    agent_state_from_sensor_pose,
    load_trajectory,
    prepare_capture_dirs,
    replay_poses,
    save_frame,
    save_intrinsics,
)

__all__ = [
    "load_config",
    "OUTSIDE",
    "ZoneScheduler",
    "zone_frame_counts",
    "zone_frame_labels",
    "apply_depth_faults",
    "apply_depth_sensor",
    "apply_lighting",
    "depth_to_vis",
    "light_exposure",
    "process_observations",
    "semantic_to_vis",
    "Engine",
    "add_start_marker",
    "make_cfg",
    "make_sensor_spec",
    "agent_state_from_sensor_pose",
    "load_trajectory",
    "prepare_capture_dirs",
    "replay_poses",
    "save_frame",
    "save_intrinsics",
]
