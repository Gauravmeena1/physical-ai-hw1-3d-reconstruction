"""Lane B unit tests: pixel pipeline + spatial ZoneScheduler (pure numpy).

Loads simulator/effects.py straight from its file path (not via the
`simulator` package) so these tests never import habitat_sim / pygame and run
outside the sim environment too.

The zone coordinates below are TEST-LOCAL SYNTHETIC GEOMETRY — a unit grid
picked to make the circle arithmetic checkable by hand. They are not, and must
not become, the zone placement for any real floor: those are placed against a
committed trajectory (HW1 specification §3.1 "Zone placement procedure").
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_EFFECTS_PATH = Path(__file__).resolve().parents[1] / "simulator" / "effects.py"
_spec = importlib.util.spec_from_file_location("_effects_under_test", _EFFECTS_PATH)
effects = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(effects)


# --- fixtures -----------------------------------------------------------------
def zone_cfg(**over):
    """Three well-separated synthetic zones on the benign/moderate/severe ladder."""
    cfg = {
        "enabled": True,
        "mode": "spatial",
        "seed": 42,                      # depth RNG only — must not reach zones
        "zones": [
            {"name": "benign",
             "center": [0.0, 0.0], "radius": 2.0,
             "flicker": {"amplitude": 0.20, "frequency": 0.8}},
            {"name": "moderate",
             "center": [10.0, 0.0], "radius": 2.0,
             "flicker": {"amplitude": 0.60, "frequency": 1.2}},
            {"name": "severe",
             "center": [0.0, -10.0], "radius": 2.0,
             "flicker": {"amplitude": 0.95, "frequency": 2.0}},
        ],
    }
    cfg.update(over)
    return cfg


DEPTH_CFG = {
    "enabled": True,
    "stuck": False,
    "noise_std": 0.05,
    "quantization": 0.05,
    "min_range": 0.3,
    "max_range": 5.0,
    "dropout_prob": 0.05,
    "light_nominal": 1.0,
    "light_noise_gain": 1.5,
    "light_dropout_gain": 0.2,
    "light_range_gain": 0.3,
}


def fixed_depth():
    return np.linspace(0.0, 6.0, 48, dtype=np.float32).reshape(6, 8)


def fixed_rgb():
    return (np.arange(4 * 5 * 3, dtype=np.uint8) * 4 % 256).reshape(4, 5, 3)


# --- zone lookup: purity (replaces the old scheduler-determinism suite) -------
def test_zone_lookup_is_pure():
    """Same (position, t) -> same override, in any order, with no hidden state.

    The temporal scheduler had a seeded lazy unroll whose realized timeline had
    to be defended against query order; ZoneScheduler has no state at all, so
    the property to defend is now purity."""
    grid = [(float(x), float(z), float(t))
            for x in np.arange(-4.0, 12.0, 1.0)
            for z in np.arange(-12.0, 4.0, 1.0)
            for t in (0.0, 0.37, 13.9)]

    ref = effects.ZoneScheduler(zone_cfg())
    answers = {q: ref.active(q[2], (q[0], q[1])) for q in grid}
    assert any(a for a in answers.values()), "fixture must hit some zone"
    assert any(not a for a in answers.values()), "fixture must miss every zone somewhere"

    # (a) a *fresh* instance answers identically -> nothing accumulated in `ref`
    fresh = effects.ZoneScheduler(zone_cfg())
    # (b) hammered in randomized order -> no order dependence
    order = np.random.default_rng(7).permutation(len(grid))
    for i in order:
        q = grid[int(i)]
        assert fresh.active(q[2], (q[0], q[1])) == answers[q]
    # (c) repeats, including revisits, still agree — and `ref` itself is unchanged
    for q in (grid[0], grid[-1], grid[0], grid[len(grid) // 2]):
        assert ref.active(q[2], (q[0], q[1])) == answers[q]
        assert fresh.active(q[2], (q[0], q[1])) == answers[q]

    # (d) each call returns a FRESH dict: mutating one must not poison the next
    ov = ref.active(0.0, (0.0, 0.0))
    ov["lighting"]["amplitude"] = 999.0
    assert ref.active(0.0, (0.0, 0.0))["lighting"]["amplitude"] == 0.20


def test_zone_lookup_is_seedless():
    """`seed` governs the depth RNG only; it must not reach the zone lookup."""
    a = effects.ZoneScheduler(zone_cfg(seed=1))
    b = effects.ZoneScheduler(zone_cfg(seed=999999))
    c = effects.ZoneScheduler({k: v for k, v in zone_cfg().items() if k != "seed"})
    for x, z in ((0.0, 0.0), (1.9, 0.0), (5.0, 5.0), (10.0, 0.5), (0.0, -10.0)):
        assert a.active(0.0, (x, z)) == b.active(0.0, (x, z)) == c.active(0.0, (x, z))


def test_outside_every_zone_is_baseline():
    s = effects.ZoneScheduler(zone_cfg())
    for pos in ((5.0, 5.0), (100.0, 100.0), (-3.0, 0.0), (0.0, 2.01), (5.0, -5.0)):
        for t in (0.0, 1.0, 7.25):
            assert s.active(t, pos) == {}


def test_hard_circular_edge_is_inclusive():
    """(x-cx)^2 + (z-cz)^2 <= r^2 — no falloff, boundary counts as inside."""
    s = effects.ZoneScheduler(zone_cfg())
    r = 2.0
    assert s.active(0.0, (r - 1e-9, 0.0)) != {}      # just inside
    assert s.active(0.0, (r, 0.0)) != {}             # exactly on the edge
    assert s.active(0.0, (r + 1e-9, 0.0)) == {}      # just outside
    # ...and the transition is a step, not a ramp: full severity right up to it
    inner = s.active(0.0, (0.0, 0.0))["lighting"]["amplitude"]
    edge = s.active(0.0, (0.0, r))["lighting"]["amplitude"]
    assert inner == edge == 0.20
    # diagonal point at exactly r
    d = r / math.sqrt(2.0)
    assert s.active(0.0, (d, d)) != {}


def test_first_match_wins_by_list_order():
    overlapping = zone_cfg(zones=[
        {"name": "first", "center": [0.0, 0.0], "radius": 3.0,
         "flicker": {"amplitude": 0.20, "frequency": 0.8}},
        {"name": "second", "center": [0.0, 0.0], "radius": 3.0,
         "flicker": {"amplitude": 0.95, "frequency": 2.0}},
    ])
    s = effects.ZoneScheduler(overlapping)
    assert s.active(0.0, (0.0, 0.0))["lighting"]["amplitude"] == 0.20

    reversed_cfg = zone_cfg(zones=list(reversed(overlapping["zones"])))
    s2 = effects.ZoneScheduler(reversed_cfg)
    assert s2.active(0.0, (0.0, 0.0))["lighting"]["amplitude"] == 0.95


def test_override_shape_and_phase_pinned_to_zero():
    s = effects.ZoneScheduler(zone_cfg())
    for name, pos, amp, freq in (("benign", (0.0, 0.0), 0.20, 0.8),
                                 ("moderate", (10.0, 0.0), 0.60, 1.2),
                                 ("severe", (0.0, -10.0), 0.95, 2.0)):
        ov = s.active(3.3, pos)
        assert set(ov) == {"lighting"}, name
        assert ov["lighting"] == {"amplitude": amp, "frequency": freq,
                                  "phase": 0.0}, name
        # phase 0 => the oscillation rides the GLOBAL clock: 1 + a*sin(2*pi*f*t)
        lighting = {"ambient_rgb": [1.0, 1.0, 1.0], "brightness": 1.0,
                    "contrast": 1.0, "gamma": 1.0, **ov["lighting"]}
        for t in (0.0, 0.31, 4.7):
            expected = 1.0 + amp * math.sin(2.0 * math.pi * freq * t)
            assert abs(effects.light_exposure(lighting, t=t) - expected) < 1e-12
        # t does not move the zone: the override is the same at every time
        assert s.active(0.0, pos) == s.active(123.4, pos)


def test_disabled_scheduler_always_baseline():
    s = effects.ZoneScheduler(zone_cfg(enabled=False))
    for pos in ((0.0, 0.0), (10.0, 0.0), (0.0, -10.0), (99.0, 99.0)):
        assert s.active(0.0, pos) == {}


def test_no_zones_is_baseline_everywhere():
    s = effects.ZoneScheduler(zone_cfg(zones=[]))
    assert s.zones == []
    assert s.active(0.0, (0.0, 0.0)) == {}


def test_rejects_temporal_mode_and_malformed_zones():
    with pytest.raises(ValueError, match="spatial"):
        effects.ZoneScheduler(zone_cfg(mode="temporal"))
    with pytest.raises(ValueError, match="center"):
        effects.ZoneScheduler(zone_cfg(zones=[{"name": "a", "center": [1.0, 2.0, 3.0],
                                               "radius": 1.0}]))
    with pytest.raises(ValueError, match="duplicate"):
        effects.ZoneScheduler(zone_cfg(zones=[
            {"name": "a", "center": [0.0, 0.0], "radius": 1.0},
            {"name": "a", "center": [9.0, 9.0], "radius": 1.0}]))
    with pytest.raises(ValueError, match="reserved"):
        effects.ZoneScheduler(zone_cfg(zones=[
            {"name": effects.OUTSIDE, "center": [0.0, 0.0], "radius": 1.0}]))
    s = effects.ZoneScheduler(zone_cfg())
    with pytest.raises(ValueError, match="position"):
        s.active(0.0, (0.0, 1.4, 0.0))              # (x, y, z) is not a position


def test_accepts_full_config_or_uncertainties_block():
    inner = zone_cfg()
    a = effects.ZoneScheduler(inner)
    b = effects.ZoneScheduler({"uncertainties": inner, "lighting": {}})
    assert a.zones == b.zones
    assert a.active(0.0, (0.0, 0.0)) == b.active(0.0, (0.0, 0.0))


# --- frames-per-zone helper (HW1 specification §3.1 step 3) ------------------------------
def _poses(xz):
    """(N,7) GT_pose.npy-shaped rows from XZ pairs (y and quaternion are junk
    the helper must ignore)."""
    out = np.zeros((len(xz), 7), dtype=np.float32)
    for i, (x, z) in enumerate(xz):
        out[i] = [x, 1.4252, z, 1.0, 0.0, 0.0, 0.0]
    return out


def test_zone_frame_counts():
    cfg = zone_cfg()
    traj = _poses([(0.0, 0.0), (1.0, 1.0),          # benign   x2
                   (10.0, 0.0),                      # moderate x1
                   (0.0, -10.0), (0.0, -9.0), (1.0, -10.0),   # severe x3
                   (5.0, 5.0), (-8.0, 3.0), (50.0, 50.0), (0.0, 2.5)])  # outside x4
    counts = effects.zone_frame_counts(cfg, traj)
    assert counts == {"benign": 2, "moderate": 1, "severe": 3, "outside": 4}
    assert sum(counts.values()) == len(traj)
    # every configured zone appears even when the trajectory misses it — that
    # zero is exactly the "zone never entered" failure the assertion guards
    miss = effects.zone_frame_counts(cfg, _poses([(50.0, 50.0)]))
    assert miss == {"benign": 0, "moderate": 0, "severe": 0, "outside": 1}


def test_zone_frame_counts_input_shapes_and_priority():
    cfg = zone_cfg()
    xz = [(0.0, 0.0), (10.0, 0.0), (5.0, 5.0)]
    expected = {"benign": 1, "moderate": 1, "severe": 0, "outside": 1}
    assert effects.zone_frame_counts(cfg, _poses(xz)) == expected           # (N,7)
    assert effects.zone_frame_counts(
        cfg, np.array([[x, 1.4, z] for x, z in xz])) == expected            # (N,3)
    assert effects.zone_frame_counts(cfg, np.array(xz)) == expected         # (N,2)
    assert effects.zone_frame_counts(cfg, np.zeros((0, 7))) == {
        "benign": 0, "moderate": 0, "severe": 0, "outside": 0}              # empty

    # labels follow the same first-match-wins order as active()
    overlapping = zone_cfg(zones=[
        {"name": "first", "center": [0.0, 0.0], "radius": 3.0,
         "flicker": {"amplitude": 0.2, "frequency": 0.8}},
        {"name": "second", "center": [1.0, 0.0], "radius": 3.0,
         "flicker": {"amplitude": 0.95, "frequency": 2.0}}])
    assert effects.zone_frame_labels(overlapping, _poses([(0.5, 0.0), (3.5, 0.0)])) \
        == ["first", "second"]
    # geometry only: `enabled: false` does not blank the counts
    assert effects.zone_frame_counts(zone_cfg(enabled=False), _poses([(0.0, 0.0)])) \
        == {"benign": 1, "moderate": 0, "severe": 0, "outside": 0}


# --- golden-array pixel tests -------------------------------------------------
def test_apply_lighting_golden():
    rgb = fixed_rgb()
    cfg = {"ambient_rgb": [1.0, 0.9, 0.7], "brightness": 1.3,
           "contrast": 1.15, "gamma": 1.8, "amplitude": 0.0}
    out = effects.apply_lighting(rgb, cfg, t=0.0)

    img = rgb.astype(np.float32) / 255.0
    img *= np.asarray(cfg["ambient_rgb"], dtype=np.float32)
    img *= cfg["brightness"]
    img = (img - 0.5) * cfg["contrast"] + 0.5
    img = np.clip(img, 0.0, 1.0) ** (1.0 / cfg["gamma"])
    expected = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

    assert out.dtype == np.uint8
    assert np.array_equal(out, expected)


def test_apply_lighting_flicker_golden():
    rgb = fixed_rgb()
    cfg = {"ambient_rgb": [1.0, 1.0, 1.0], "brightness": 1.0, "contrast": 1.0,
           "gamma": 1.0, "amplitude": 0.5, "frequency": 2.0, "phase": 0.3}
    t = 0.37
    out = effects.apply_lighting(rgb, cfg, t=t)

    osc = 1.0 + 0.5 * np.sin(2.0 * np.pi * 2.0 * t + 0.3)
    img = rgb.astype(np.float32) / 255.0 * osc
    expected = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

    assert np.array_equal(out, expected)
    assert abs(effects.light_exposure(cfg, t=t) - osc) < 1e-12


def test_apply_depth_faults_golden():
    depth = fixed_depth()
    out = effects.apply_depth_faults(depth, DEPTH_CFG, np.random.default_rng(9))

    rr = np.random.default_rng(9)  # replay the exact draw sequence
    d = depth.astype(np.float32).copy()
    d += rr.normal(0.0, DEPTH_CFG["noise_std"], size=d.shape).astype(np.float32)
    step = DEPTH_CFG["quantization"]
    d = np.round(d / step) * step
    d[(d < DEPTH_CFG["min_range"]) | (d > DEPTH_CFG["max_range"])] = 0.0
    d[rr.random(d.shape) < DEPTH_CFG["dropout_prob"]] = 0.0
    expected = np.clip(d, 0.0, None)

    assert np.array_equal(out, expected)


def test_apply_depth_sensor_deterministic_with_fixed_rng():
    depth = fixed_depth()
    out1 = effects.apply_depth_sensor(depth, DEPTH_CFG, 1.7, np.random.default_rng(123))
    out2 = effects.apply_depth_sensor(depth, DEPTH_CFG, 1.7, np.random.default_rng(123))
    assert np.array_equal(out1, out2)
    # light coupling actually changes the output at nominal vs stressed exposure
    nominal = effects.apply_depth_sensor(depth, DEPTH_CFG, 1.0, np.random.default_rng(123))
    assert not np.array_equal(out1, nominal)


def test_per_frame_rng_stream_reproducible():
    # Engine keys the per-frame stream as default_rng([seed, ms]) — the same key
    # must give bit-identical depth faults regardless of the window schedule.
    depth = fixed_depth()
    seed, ms = 42, 1234
    f1 = effects.apply_depth_faults(depth, DEPTH_CFG, np.random.default_rng([seed, ms]))
    f2 = effects.apply_depth_faults(depth, DEPTH_CFG, np.random.default_rng([seed, ms]))
    assert np.array_equal(f1, f2)
    f3 = effects.apply_depth_faults(depth, DEPTH_CFG, np.random.default_rng([seed, ms + 1]))
    assert not np.array_equal(f1, f3)


def test_depth_faults_disabled_or_stuck_returns_zeros():
    depth = fixed_depth()
    rng = np.random.default_rng(0)
    assert np.array_equal(effects.apply_depth_faults(depth, {**DEPTH_CFG, "enabled": False}, rng),
                          np.zeros_like(depth))
    assert np.array_equal(effects.apply_depth_faults(depth, {**DEPTH_CFG, "stuck": True}, rng),
                          np.zeros_like(depth))


def test_depth_to_vis():
    depth = np.array([[0.0, 2.5], [5.0, 10.0]], dtype=np.float32)
    vis = effects.depth_to_vis(depth, 5.0)
    assert vis.shape == (2, 2, 3)
    assert vis.dtype == np.uint8
    assert np.array_equal(vis[:, :, 0], np.array([[0, 127], [255, 255]], dtype=np.uint8))
    assert np.array_equal(vis[:, :, 0], vis[:, :, 1])
    assert np.array_equal(vis[:, :, 0], vis[:, :, 2])


def test_no_global_np_random_in_module():
    src = _EFFECTS_PATH.read_text()
    for banned in ("np.random.normal", "np.random.random", "np.random.seed",
                   "np.random.uniform", "np.random.choice"):
        assert banned not in src, f"global RNG call {banned} must not appear in effects.py"


# --- performance paths must not change pixels --------------------------------
def test_apply_lighting_lut_matches_dense_pipeline():
    """apply_lighting is a 256-entry LUT over _apply_lighting_dense; it must be
    bit-identical to running the dense pipeline on every pixel, for tinted,
    flickering, gamma'd and neutral configs alike, RGBA input included."""
    rng = np.random.default_rng(0)
    rgba = rng.integers(0, 256, size=(37, 41, 4), dtype=np.uint8)   # habitat colour is RGBA
    rgba[0, :4, :3] = [[0, 0, 0], [255, 255, 255], [1, 254, 128], [127, 128, 129]]
    cfgs = [
        dict(brightness=1.0, contrast=1.0, gamma=1.0, ambient_rgb=[1.0, 1.0, 1.0]),
        dict(brightness=1.3, contrast=1.2, gamma=2.2, ambient_rgb=[1.0, 0.9, 0.7],
             amplitude=0.6, frequency=1.17, phase=0.0),
        dict(brightness=0.4, contrast=0.8, gamma=0.7, ambient_rgb=[0.8, 1.0, 1.1]),
        dict(brightness=1.95, contrast=1.0, gamma=1.0, ambient_rgb=[1.0, 1.0, 1.0],
             amplitude=0.95, frequency=1.91, phase=0.0),
    ]
    for cfg in cfgs:
        for t in (0.0, 0.37, 12.5):
            fast = effects.apply_lighting(rgba, cfg, t)
            ref = effects._apply_lighting_dense(rgba, cfg, t)
            assert fast.shape == (37, 41, 3) and fast.dtype == np.uint8
            np.testing.assert_array_equal(fast, ref)
    # 3-channel input works too (the unit tests' fixed_rgb layout)
    np.testing.assert_array_equal(
        effects.apply_lighting(fixed_rgb(), cfgs[1], 0.5),
        effects._apply_lighting_dense(fixed_rgb(), cfgs[1], 0.5))


def test_process_observations_tolerates_missing_optional_sensors():
    """engine.make_cfg leaves semantic / bird's-eye sensors out when nothing
    consumes them; the frame then carries None for those keys and everything
    else is unaffected."""
    config = {"lighting": dict(brightness=1.0, contrast=1.0, gamma=1.0,
                               ambient_rgb=[1.0, 1.0, 1.0]),
              "depth": DEPTH_CFG}
    rgba = np.zeros((6, 8, 4), dtype=np.uint8)
    obs = {"color_sensor": rgba, "depth_sensor": fixed_depth()}
    frame = effects.process_observations(obs, config, 0.0, np.random.default_rng(1))
    assert frame["birdseye"] is None and frame["semantic"] is None
    assert frame["rgb"].shape == (6, 8, 3)
    assert frame["depth_m"].shape == (6, 8) and frame["depth_vis"].shape == (6, 8, 3)
    full = effects.process_observations(
        {**obs, "birdseye_sensor": rgba}, config, 0.0, np.random.default_rng(1))
    assert full["birdseye"].shape == (6, 8, 3)
    np.testing.assert_array_equal(full["depth_m"], frame["depth_m"])


def test_lighting_and_depth_vis_numpy_fallback_matches_cv2(monkeypatch):
    """With cv2 unavailable the numpy gathers must give the same pixels as the
    cv2 paths (and both the same as the dense reference)."""
    if effects._cv2 is None:
        pytest.skip("cv2 not importable here; only the numpy path exists")
    rng = np.random.default_rng(3)
    rgba = rng.integers(0, 256, size=(23, 29, 4), dtype=np.uint8)
    depth = rng.random((23, 29), dtype=np.float32) * 12.0
    cfg = dict(brightness=1.1, contrast=1.2, gamma=1.8, ambient_rgb=[1.0, 0.9, 0.8],
               amplitude=0.3, frequency=0.73, phase=0.0)
    grey = dict(brightness=0.7, contrast=1.0, gamma=1.0, ambient_rgb=[1.0, 1.0, 1.0])
    with_cv2 = [effects.apply_lighting(rgba, cfg, 0.2), effects.apply_lighting(rgba, grey, 0.2),
                effects.depth_to_vis(depth, 10.0)]
    monkeypatch.setattr(effects, "_cv2", None)
    without = [effects.apply_lighting(rgba, cfg, 0.2), effects.apply_lighting(rgba, grey, 0.2),
               effects.depth_to_vis(depth, 10.0)]
    for a, b in zip(with_cv2, without):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(without[0], effects._apply_lighting_dense(rgba, cfg, 0.2))
    np.testing.assert_array_equal(without[1], effects._apply_lighting_dense(rgba, grey, 0.2))
