"""Simulator pipeline e2e (HW1 specification "test_e2e.py spec" — the REAL integration gate).

Cases: replay smoke, determinism, baseline invariance OUTSIDE zones, effects
firing INSIDE zones, hard-error trajectory paths, and headless/no-pygame.

FIXTURE LIMITATION — READ BEFORE ADDING A CASE. The committed 10-pose fixture is
a pure IN-PLACE ROTATION: all ten poses sit at the same world XZ (only the
quaternion changes). Under the spatial regime (HW1 specification §3.1) zone membership is
a function of position alone, so this fixture CANNOT split one run into in-zone
and out-of-zone frames — every frame is inside, or every frame is outside,
depending on where the zone is put. The two branches are therefore exercised as
two separate runs (`outside` / `inside`), and the interesting mixed-membership
logic — first match wins, hard edges, frames-per-zone counting — is covered
exhaustively in the pure-numpy unit tests (tests/test_effects.py). Give this
suite a moving fixture and the two runs can be merged back into one.

FAIL-LOUD policy: a missing Replica scene or pose fixture ABORTS collection with
an actionable message (run `pixi run fetch-replica`) instead of skipping — a
skipped suite must never read as green. Escape hatch for machines that
legitimately lack the scene: SIM_E2E_SKIP=1 (skips loudly).

Run: env -u PYTHONPATH pixi run -e habitat python -m pytest packages/simulator/tests/
Runtime target < 2 min: committed 10-pose fixture
(`tests/fixtures/mini_secondfloor.npy`), 128x128 sensors.
"""

import copy
import json
import math
import os
import sys
from pathlib import Path

# Headless: drop DISPLAY before ANY Engine construction — proves the EGL
# offscreen path works with no X display at all (Engine tolerates unset DISPLAY).
os.environ.pop("DISPLAY", None)

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Project-root resolution (tests may be invoked from any cwd): support both the
# monorepo checkout (outer pixi.toml + hw1/) and this standalone hw1 workspace
# (pixi.toml + configs/). packages/simulator ships its own pixi.toml, so matching
# on that alone would land too deep and resolve scene/config paths incorrectly.
# ---------------------------------------------------------------------------
def _find_repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "pixi.toml").is_file() and (parent / "hw1").is_dir():
            return parent
        if ((parent / "pixi.toml").is_file()
                and (parent / "configs" / "second_floor.yaml").is_file()):
            return parent
    raise RuntimeError(
        "could not locate project root (no suitable pixi.toml above %s)" % __file__)


REPO = _find_repo_root()
HW1 = REPO / "hw1" if (REPO / "hw1").is_dir() else REPO
SCENE = HW1 / "replica_v1" / "apartment_0" / "habitat" / "mesh_semantic.ply"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_secondfloor.npy"
CONFIG_YAML = HW1 / "configs" / "second_floor.yaml"
FPS = 30.0
N_POSES = 10

# --- fail-loud gate (module import/collection time) ------------------------
if os.environ.get("SIM_E2E_SKIP") == "1":
    pytest.skip(
        "SIM_E2E_SKIP=1: simulator e2e suite SKIPPED on request — this machine "
        "claims to legitimately lack the Replica scene. The pipeline was NOT "
        "exercised.",
        allow_module_level=True,
    )

_missing = [str(p) for p in (SCENE, FIXTURE, CONFIG_YAML) if not p.exists()]
if _missing:
    raise pytest.UsageError(
        "simulator e2e prerequisites missing:\n  "
        + "\n  ".join(_missing)
        + "\nrun `pixi run fetch-replica` (scene) / restore tests/fixtures "
        "(fixture). Missing assets are a HARD ERROR, not a skip "
        "(set SIM_E2E_SKIP=1 only on machines that legitimately lack the scene)."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fixture_xz():
    """The fixture's world XZ. All ten poses share it — see the module note."""
    poses = np.load(FIXTURE)
    xz = poses[:, [0, 2]]
    assert np.allclose(xz, xz[0]), (
        "the fixture moved! it is no longer a pure in-place rotation, so the "
        "one-membership-per-run workaround in this module can be replaced by a "
        "single mixed run — see the module docstring")
    return float(xz[0][0]), float(xz[0][1])


# Test-local zone geometry, derived from the fixture at runtime — never
# hardcoded scene coordinates. FLICKER_HZ is chosen so that sin(2*pi*f*i/30) is
# nonzero for every captured frame i in 1..10 (it peaks at i=10): with the phase
# pinned at 0 (HW1 specification §3.1) a frame whose sine happens to vanish is
# bit-identical to baseline even inside a zone.
FLICKER_HZ = 0.75
FLICKER_AMPLITUDE = 0.9
ZONE_RADIUS = 1.0


def _zone_uncertainties(covers_fixture, enabled=True):
    """A spatial `uncertainties` block whose single zone either contains the
    fixture position or sits far away from it."""
    x, z = _fixture_xz()
    center = [x, z] if covers_fixture else [x + 50.0, z + 50.0]
    return {
        "enabled": enabled,
        "mode": "spatial",
        "seed": 20,
        "zones": [{"name": "severe_test",
                   "center": center,
                   "radius": ZONE_RADIUS,
                   "flicker": {"amplitude": FLICKER_AMPLITUDE,
                               "frequency": FLICKER_HZ}}],
    }


def _base_config(covers_fixture=False):
    """second_floor.yaml with test-local overrides: fixture trajectory,
    absolute scene path, 128x128 sensors (speed — allowed per spec), and a
    test-local spatial `uncertainties` block (this suite must not depend on
    where the shipped config happens to place its zones)."""
    from simulator import load_config

    cfg = load_config(str(CONFIG_YAML))
    cfg["scene"]["path"] = str(SCENE)
    cfg["trajectory"] = str(FIXTURE)
    cfg["camera"].update(width=128, height=128)
    cfg["birdseye"].update(width=128, height=128)
    cfg["uncertainties"] = _zone_uncertainties(covers_fixture)
    return cfg


def _gain(i):
    """Exposure gain the flicker zone applies to capture frame i (1-indexed):
    phase is pinned to 0, so the sine reads the global clock t = i / FPS."""
    return 1.0 + FLICKER_AMPLITUDE * math.sin(2.0 * math.pi * FLICKER_HZ * i / FPS)


def _run_replay(cfg, scheduler_on, out_dir):
    """One Engine + replay_poses pass over the fixture -> rgb/ depth/
    GT_pose.npy / intrinsics.json. Returns the captured (N,7) poses.

    One habitat Simulator per process at a time: the Engine is always closed
    (finally) before the caller constructs the next one."""
    from simulator import (Engine, ZoneScheduler, load_trajectory,
                           prepare_capture_dirs, replay_poses, save_frame)

    poses = load_trajectory(cfg["trajectory"])
    out_dir = Path(out_dir)
    run_cfg = copy.deepcopy(cfg)
    run_cfg["output"] = dict(cfg["output"], save_rgb=True, save_depth=True,
                             save_semantic=False, clear_existing=False)
    prepare_capture_dirs(run_cfg, str(out_dir))   # dirs + intrinsics.json
    scheduler = ZoneScheduler(run_cfg["uncertainties"]) if scheduler_on else None

    engine = Engine(run_cfg, scheduler=scheduler, fps_nominal=FPS)
    try:
        def out_cb(frame, sensor_state, idx):
            save_frame(frame, sensor_state, str(out_dir), run_cfg["output"], idx)

        captured = replay_poses(engine, poses, out_cb)
        np.save(out_dir / "GT_pose.npy", np.asarray(captured, dtype=np.float32))
    finally:
        engine.close()
    return captured


def _png(root, kind, idx):
    return Path(root) / kind / f"{idx}.png"


def _read_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Module-scoped replay runs (Engines constructed sequentially, each closed
# before the next — determinism/invariance need fresh Engines by design).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e_runs")
    cfg_out = _base_config(covers_fixture=False)   # zone 50 m away
    cfg_in = _base_config(covers_fixture=True)     # zone on the fixture
    out = {"cfg_outside": cfg_out, "cfg_inside": cfg_in, "root": root}

    out["baseline"] = root / "baseline"          # scheduler OFF
    out["cap_baseline"] = _run_replay(cfg_out, False, out["baseline"])

    out["outside"] = root / "outside"            # zones ON, every frame outside
    out["cap_outside"] = _run_replay(cfg_out, True, out["outside"])

    out["inside1"] = root / "inside1"            # zones ON, every frame inside
    out["cap_inside1"] = _run_replay(cfg_in, True, out["inside1"])

    out["inside2"] = root / "inside2"            # identical rerun (determinism)
    out["cap_inside2"] = _run_replay(cfg_in, True, out["inside2"])
    return out


# ---------------------------------------------------------------------------
# 1. replay_smoke
# ---------------------------------------------------------------------------
def test_replay_smoke(runs):
    run = runs["inside1"]
    for i in range(1, N_POSES + 1):
        assert _png(run, "rgb", i).is_file(), f"missing rgb/{i}.png"
        assert _png(run, "depth", i).is_file(), f"missing depth/{i}.png"

    gt = np.load(run / "GT_pose.npy")
    assert gt.shape == (N_POSES, 7)

    fixture = np.load(FIXTURE)
    # Teleport replay is exact: captured sensor poses match the fixture input
    # (empirically max abs diff ~6e-8; loose-but-meaningful tolerance).
    assert np.allclose(gt, fixture, atol=1e-5), (
        "captured poses drifted from fixture input "
        f"(max abs diff {np.abs(gt - fixture).max():.3e})")

    # Every capture carries its own camera parameters and NOTHING else about
    # the camera (HW1 specification D5) — no extrinsics, no uncertainty, no coupling.
    cam = runs["cfg_inside"]["camera"]
    for cond in ("baseline", "outside", "inside1"):
        intr = _read_json(Path(runs[cond]) / "intrinsics.json")
        assert set(intr) == {"width", "height", "hfov"}, f"{cond}: {intr}"
        assert intr == {"width": int(cam["width"]), "height": int(cam["height"]),
                        "hfov": float(cam["hfov"])}, cond
    # ...and no window ground truth is emitted any more (HW1 specification §3.1/D6).
    assert not (run / "windows.json").exists()


# ---------------------------------------------------------------------------
# 2. determinism
# ---------------------------------------------------------------------------
def test_determinism(runs):
    a, b = runs["inside1"], runs["inside2"]
    for kind in ("rgb", "depth"):
        for i in range(1, N_POSES + 1):
            assert _png(a, kind, i).read_bytes() == _png(b, kind, i).read_bytes(), (
                f"same config+seed produced different {kind}/{i}.png")
    assert np.array_equal(runs["cap_inside1"], runs["cap_inside2"])


# ---------------------------------------------------------------------------
# 3. baseline_invariance OUTSIDE zones (per-frame-RNG invariant: frames in no
#    zone are bit-identical between scheduler ON and OFF runs)
# ---------------------------------------------------------------------------
def test_baseline_invariance_outside_zones(runs):
    from simulator import zone_frame_counts

    # The zone sits 50 m from the (stationary) fixture, so no frame is in it —
    # frames-per-zone assertion, HW1 specification §3.1 step 3.
    counts = zone_frame_counts(runs["cfg_outside"], runs["cap_outside"])
    assert counts == {"severe_test": 0, "outside": N_POSES}, counts

    for kind in ("rgb", "depth"):
        for i in range(1, N_POSES + 1):
            assert (_png(runs["baseline"], kind, i).read_bytes()
                    == _png(runs["outside"], kind, i).read_bytes()), (
                f"{kind}/{i}.png differs between scheduler ON/OFF despite the "
                f"agent standing outside every zone")


def test_effects_fire_inside_zones(runs):
    from simulator import zone_frame_counts

    # Same trajectory, zone moved onto it: now every frame is in the zone.
    counts = zone_frame_counts(runs["cfg_inside"], runs["cap_inside1"])
    assert counts == {"severe_test": N_POSES, "outside": 0}, counts

    for i in range(1, N_POSES + 1):
        assert (_png(runs["baseline"], "rgb", i).read_bytes()
                != _png(runs["inside1"], "rgb", i).read_bytes()), (
            f"in-zone rgb/{i}.png identical to baseline — effect did not fire "
            f"(exposure gain {_gain(i):.3f})")


# ---------------------------------------------------------------------------
# 4. effects_fire, quantitatively: at the flicker peak the frame is BRIGHTER,
#    and the light->depth coupling drives dropout up on that same frame.
# ---------------------------------------------------------------------------
def test_effects_fire_photometry_and_depth_coupling(runs):
    import cv2

    # Phase is pinned at 0, so the gain is known per frame; take the peak.
    i = max(range(1, N_POSES + 1), key=_gain)
    gain = _gain(i)
    assert gain > 1.3, f"fixture/frequency no longer reach a strong peak ({gain:.3f})"

    rgb_base = cv2.imread(str(_png(runs["baseline"], "rgb", i)))
    rgb_zone = cv2.imread(str(_png(runs["inside1"], "rgb", i)))
    assert rgb_base is not None and rgb_zone is not None
    assert rgb_zone.mean() > rgb_base.mean(), (
        f"in-zone frame {i} not brighter at gain {gain:.3f}: "
        f"{rgb_zone.mean():.2f} vs baseline {rgb_base.mean():.2f}")

    d_base = cv2.imread(str(_png(runs["baseline"], "depth", i)), cv2.IMREAD_UNCHANGED)
    d_zone = cv2.imread(str(_png(runs["inside1"], "depth", i)), cv2.IMREAD_UNCHANGED)
    assert d_base is not None and d_zone is not None and d_base.dtype == np.uint16
    frac_base = float(np.mean(d_base == 0))
    frac_zone = float(np.mean(d_zone == 0))
    # Light coupling: stress = |gain - 1| ~= 0.9 -> dropout_prob ~= 0.27
    # (light_dropout_gain 0.3) on top of a dropout-free baseline, plus a
    # shortened max_range.
    assert frac_zone > frac_base + 0.10, (
        f"light-coupled depth dropout did not fire: zero-frac {frac_zone:.3f} "
        f"(in zone) vs {frac_base:.3f} (baseline)")


# ---------------------------------------------------------------------------
# 5. missing_trajectory_raises (replay API — hard error, no [skip])
# ---------------------------------------------------------------------------
def test_missing_trajectory_raises(tmp_path):
    from simulator import load_trajectory

    with pytest.raises(FileNotFoundError, match="trajectory not found"):
        load_trajectory(str(tmp_path / "does_not_exist.npy"))


# ---------------------------------------------------------------------------
# 6. json_trajectory_raises (action replay deprecated; pointer to .npy)
# ---------------------------------------------------------------------------
def test_json_trajectory_raises():
    from simulator import load_trajectory

    with pytest.raises(ValueError, match=r"\.npy"):
        load_trajectory("x.json")


# ---------------------------------------------------------------------------
# 7. headless — must stay LAST in this module: after every Engine run above,
#    pygame was never imported (viewer is opt-in; nothing here touches it).
# ---------------------------------------------------------------------------
def test_z_no_pygame(runs):
    assert "DISPLAY" not in os.environ
    assert "pygame" not in sys.modules, (
        "pygame was imported during the e2e suite — the headless pipeline must "
        "never touch simulator.viewer")
