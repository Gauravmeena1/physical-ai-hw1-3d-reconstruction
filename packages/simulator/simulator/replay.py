"""Capture I/O + trajectory replay: .npy (N,7) sensor-pose teleport replay ONLY.

JSON action trajectories are unsupported because they reproduce a path only when
the config's actuation matches the values that generated them. load_trajectory
raises on .json with a pointer to .npy pose replay.

Contract highlights (HW1 specification):
- replay_poses returns the (N,7) captured poses; the CALLER saves GT_pose.npy.
- Missing/invalid trajectory is a hard error, never a silent skip.
- prepare_capture_dirs is the ONE capture-writing entry point used by load.py,
  so every capture directory carries
  its own intrinsics.json (HW1 specification D5).
"""

import json
import os
import shutil

import numpy as np
import cv2

from scipy.spatial.transform import Rotation as Rot

import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs


def load_trajectory(path):
    """Return the (N,7) pose array [x,y,z, qw,qx,qy,qz] from a .npy trajectory.

    JSON action trajectories only reproduce a path when the config's actuation
    matches the values that generated them. A .json path raises with a pointer
    to .npy pose replay; a missing path raises FileNotFoundError."""
    if path.endswith(".json"):
        raise ValueError(
            f"action replay (.json) is no longer supported: {path} — "
            "use the .npy (N,7) sensor-pose trajectory instead (exact pose "
            "replay via replay_poses, independent of config actuation)")
    if not os.path.exists(path):
        raise FileNotFoundError(f"trajectory not found: {path}")
    if path.endswith(".npy"):
        return np.load(path)
    raise ValueError(f"unsupported trajectory file (want .npy): {path}")


def save_intrinsics(config, data_root):
    """Write <data_root>/intrinsics.json = {"width", "height", "hfov"}.

    EXACTLY those three keys, taken from config["camera"] (hfov in DEGREES).
    They are the only camera facts a capture ships: without them
    `depth_image_to_point_cloud` is unimplementable against a dataset that ships
    with no config (HW1 specification D5), and they reveal nothing about how the
    environment degrades — no extrinsics, no uncertainty zones, no depth
    coupling. Every capture carries its own, so reconstruction reads the
    intrinsics of the capture it is reconstructing and cannot unproject one
    floor's depth through another floor's camera. Returns the path written."""
    cam = config["camera"]
    path = os.path.join(data_root, "intrinsics.json")
    with open(path, "w") as f:
        json.dump({"width": int(cam["width"]),
                   "height": int(cam["height"]),
                   "hfov": float(cam["hfov"])}, f, indent=2)
    return path


def prepare_capture_dirs(config, data_root):
    """Create a capture directory and emit its intrinsics.json.

    Makes <data_root>/{rgb,depth} (+ semantic/ when output.save_semantic), then
    writes intrinsics.json next to where GT_pose.npy will land. This is the
    shared capture path — load.py (both interactive and replay collection) goes
    through it, so no
    tool can produce a capture without its intrinsics.

    WARNING: output.clear_existing deletes the whole `data_root` tree first."""
    out = config["output"]
    if out.get("clear_existing", False) and os.path.isdir(data_root):
        shutil.rmtree(data_root)
    subs = ["rgb", "depth"]
    if out.get("save_semantic", False):
        subs.append("semantic")
    for sub in subs:
        os.makedirs(os.path.join(data_root, sub), exist_ok=True)
    return save_intrinsics(config, data_root)


def agent_state_from_sensor_pose(pose, config):
    """Build an AgentState that puts the COLOR sensor at world pose `pose`
    ([x,y,z, qw,qx,qy,qz]). Inverts the fixed camera extrinsic (config.camera
    position + orientation) so the teleported sensor matches the pose exactly."""
    cam = config["camera"]
    sensor_R = Rot.from_quat([pose[4], pose[5], pose[6], pose[3]])   # x,y,z,w
    cam_R = Rot.from_euler("xyz", [float(v) for v in cam["orientation"]])
    t_cam = np.asarray(cam["position"], dtype=np.float64)

    # world_sensor = agent ∘ extrinsic  =>  agent_R = sensor_R · cam_R⁻¹,
    # agent_pos = sensor_pos − agent_R · t_cam.
    agent_R = sensor_R * cam_R.inv()
    agent_pos = np.asarray(pose[:3], dtype=np.float64) - agent_R.apply(t_cam)

    st = habitat_sim.AgentState()
    st.position = agent_pos.astype(np.float32)
    st.rotation = quat_from_coeffs(agent_R.as_quat().astype(np.float32))   # [x,y,z,w]
    return st


def save_frame(frame, sensor_state, data_root, out, idx):
    """Write one capture's rgb/depth/semantic PNGs and return its GT pose row
    [x, y, z, qw, qx, qy, qz] from the color-sensor world pose."""
    if out["save_rgb"]:
        cv2.imwrite(os.path.join(data_root, "rgb", f"{idx}.png"),
                    frame["rgb"][:, :, ::-1])              # RGB -> BGR for cv2
    if out["save_depth"]:
        # 16-bit millimetres (NOT the 8-bit preview): preserves the injected
        # depth noise / coupling so the reconstructor sees it instead of it being
        # swamped by 8-bit quantisation. utils.load_depth_meters detects uint16.
        depth_mm = np.clip(frame["depth_m"] * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(data_root, "depth", f"{idx}.png"), depth_mm)
    if out["save_semantic"]:
        cv2.imwrite(os.path.join(data_root, "semantic", f"{idx}.png"),
                    frame["semantic"][:, :, ::-1])
    p, r = sensor_state.position, sensor_state.rotation
    return [p[0], p[1], p[2], r.w, r.x, r.y, r.z]


def replay_poses(engine, poses, out_cb):
    """Teleport the camera to each pose, capture a frame. Exact replay.

    Per frame i (1-based): the agent is teleported so the color sensor matches
    poses[i-1], the frame is processed via engine.observe(i / engine.fps_nominal)
    — frame index over nominal fps is replay's only deterministic clock, so
    flicker/windows/depth noise are reproducible across runs and machines —
    and `out_cb(frame, sensor_state, i)` is invoked. out_cb owns all output I/O
    (e.g. wrap save_frame, or preview); returning False aborts the replay early.

    Returns the (N,7) captured poses [x,y,z, qw,qx,qy,qz]; the CALLER saves
    GT_pose.npy."""
    cam_extr = []
    for i, pose in enumerate(poses, start=1):
        engine.agent.set_state(agent_state_from_sensor_pose(pose, engine.config))
        frame = engine.observe(i / engine.fps_nominal)
        sensor_state = engine.agent.get_state().sensor_states["color_sensor"]
        p, r = sensor_state.position, sensor_state.rotation
        cam_extr.append([p[0], p[1], p[2], r.w, r.x, r.y, r.z])
        if i % 25 == 0 or i == len(poses):
            print(f"replay: captured {i}/{len(poses)} frames")
        if out_cb(frame, sensor_state, i) is False:
            print(f"replay: aborted by user at frame {i}")
            break
    return np.asarray(cam_extr, dtype=np.float32)
