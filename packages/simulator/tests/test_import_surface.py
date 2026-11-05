"""P0 golden test: the public import surface every lane must keep green.

Asserts importability only — bodies may raise NotImplementedError until lanes land.
Also asserts that importing the package does NOT pull in pygame (viewer is opt-in).
"""

import sys
from pathlib import Path


def test_import_surface():
    import simulator
    from simulator import (  # noqa: F401
        OUTSIDE,
        Engine,
        ZoneScheduler,
        add_start_marker,
        agent_state_from_sensor_pose,
        apply_depth_faults,
        apply_depth_sensor,
        apply_lighting,
        depth_to_vis,
        light_exposure,
        load_config,
        load_trajectory,
        make_cfg,
        make_sensor_spec,
        prepare_capture_dirs,
        process_observations,
        replay_poses,
        save_frame,
        save_intrinsics,
        semantic_to_vis,
        zone_frame_counts,
        zone_frame_labels,
    )

    assert simulator.__all__


def test_temporal_scheduler_is_gone():
    """UncertaintyScheduler / windows.json were deleted (HW1 specification §3.1) — a
    leftover import path would let a stale caller silently keep the temporal
    regime alive."""
    import simulator

    assert not hasattr(simulator, "UncertaintyScheduler")
    assert "UncertaintyScheduler" not in simulator.__all__
    for path in sorted(Path(simulator.__file__).parent.glob("*.py")):
        src = path.read_text()
        # prose may name the class it replaced; a definition or call may not
        assert "class UncertaintyScheduler" not in src, path.name
        assert "UncertaintyScheduler(" not in src, path.name
        # a windows.json path would appear as a string LITERAL (prose mentioning
        # its removal is fine)
        for literal in ('"windows.json"', "'windows.json'"):
            assert literal not in src, f"{path.name} still writes {literal}"


def test_no_pygame_on_package_import():
    assert "pygame" not in sys.modules, (
        "importing `simulator` must not import pygame; only `simulator.viewer` may"
    )
