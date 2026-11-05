"""Config loading: yaml + schema defaults merge.

NOTE (contract): the defaults table covers the `uncertainties` block and every
lighting/depth key consumed by the simulator. See HW1 specification "Interface
contract".
"""

import copy

import yaml

# Every key the pixel pipeline reads gets a default here. A sparse YAML can
# therefore state only what it changes; file values always win over defaults
# (deep merge).
#
# Defaults are chosen to be inert: neutral lighting, fault-free depth, zero
# light-coupling gains — merging them into a config never changes behavior the
# config already specifies.
_DEFAULTS = {
    "lighting": {
        "brightness": 1.0,        # exposure gain (1.0 = neutral)
        "contrast": 1.0,          # contrast around mid-grey
        "gamma": 1.0,             # 1.0 = no gamma correction
        "ambient_rgb": [1.0, 1.0, 1.0],  # colour tint / temperature
        "amplitude": 0.0,         # flicker amplitude (0 = steady)
        "frequency": 0.0,         # flicker Hz
        "phase": 0.0,             # flicker phase (rad)
    },
    "depth": {
        "enabled": True,
        "stuck": False,           # dead sensor -> all zeros
        "noise_std": 0.0,         # gaussian noise sigma (m)
        "quantization": 0.0,      # step size (m); 0 = off
        "min_range": 0.0,         # readings below return 0
        "max_range": 10.0,        # readings above return 0
        "dropout_prob": 0.0,      # random per-pixel dropout
        "redwood": False,         # habitat in-sim Redwood noise model
        "redwood_multiplier": 1.0,
        # --- ambient-light coupling (stress = |exposure - light_nominal|) ---
        "light_nominal": 1.0,
        "light_noise_gain": 0.0,
        "light_dropout_gain": 0.0,
        "light_range_gain": 0.0,
    },
    # SPATIAL uncertainty injection (HW1 specification §3.1): flicker zones keyed on WHERE
    # the agent is, not on when it got there. A stateless, seedless ZoneScheduler
    # looks the agent's XZ position up in `zones` (first match wins, hard
    # circular edges). `enabled: false` (evaluate's baseline run) skips the
    # scheduler entirely.
    "uncertainties": {
        "enabled": True,
        "mode": "spatial",        # the only mode; the temporal scheduler is gone
        "seed": 42,               # ONLY the per-frame depth RNG (engine.observe)
        # Inert default: no zones -> nothing ever fires. Real zone centres are
        # placed per floor against the committed trajectory (HW1 specification §3.1
        # "Zone placement procedure"), never defaulted here — a zone the
        # trajectory misses degenerates the mixed run to baseline.
        "zones": [],
    },
}


def _deep_merge(base, override):
    """Recursively merge `override` into `base` in place; override wins.

    Nested dicts merge key-by-key (so a config may state only the keys it
    changes, e.g. one depth knob); any non-dict value replaces the default
    wholesale — LISTS INCLUDED, so a config's `uncertainties.zones` list
    replaces the (empty) default outright and is never element-merged."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path):
    """Load a YAML config and merge schema defaults. Returns a dict.

    File values win over defaults; defaults only fill gaps (deep merge, see
    _DEFAULTS). Blocks with no defaults (scene, agent, camera, ...) pass
    through untouched."""
    with open(path, "r") as f:
        loaded = yaml.safe_load(f) or {}
    return _deep_merge(copy.deepcopy(_DEFAULTS), loaded)
