# `load.py` — Habitat data collector

> **TA-internal implementation documentation.** This repository contains the
> complete homework implementation and reference materials.

Interactive Habitat-Sim collector for hw1. A thin driver over the
`simulator` package: it parses the CLI, wires the pieces together, and runs
the event loop. All simulation, pixel-pipeline, replay, and viewer logic
lives in `packages/simulator`.

## Running it

Interactive collection (pygame window, keyboard-driven):

```bash
pixi run -e habitat python load.py
```

Replay a saved pose trajectory through the same frame-saving pipeline:

```bash
pixi run -e habitat python load.py --trajectory trajectories/secondfloor.npy
```

The config defaults to `configs/second_floor.yaml` (`--config` to
override); `--output-root` overrides `output.root`; `--fps` paces the preview
loop. The config's `trajectory` entry documents the course replay input but is
not selected implicitly: pass `--trajectory` explicitly so an omitted argument
still means interactive collection.

## Controls

| Keys | Action |
|---|---|
| `w` / `s` | Move forward / backward |
| `a` / `d` | Turn left / right |
| `c` / `SPACE` | Capture the current frame |
| `q` / `ESC` | Finish and quit (aborts a replay too) |

## Uncertainty zones are spatial

The config's `uncertainties` block defines hard-edged circular flicker zones
(`center: [x, z]`, `radius`). A frame is degraded iff the **agent stands
inside a zone** — where it is, not when it got there. With a fixed `.npy` pose
replay, fixed `--fps`, and the configured depth seed, zone membership, flicker
phase, and depth-noise sequence are reproducible. Interactive collection uses
wall-clock time and human input, so it is intentionally not bit-for-bit
reproducible. `--clean` (or `uncertainties.enabled: false`) disables spatial
uncertainty zones. It leaves the configured baseline depth sensor model,
including `depth.noise_std`, in effect; call this a zone-free capture rather
than noiseless data. After an enabled-zone run, a per-zone frame count is
printed. A zone never entered contributed no perturbed frames.

## Outputs (under `output.root`)

- `rgb/<n>.png`, `depth/<n>.png`, optionally `semantic/<n>.png`
- `GT_pose.npy`: `(N, 7)` poses `[x, y, z, qw, qx, qy, qz]`
- `intrinsics.json`: `{"width", "height", "hfov"}` — the capture's own camera
  parameters. Reconstruction reads them from the capture being
  reconstructed, never from a config.

## Performance notes

The preview holds 30 fps by engineering out the expensive parts: the raw
readout is cached while the agent stands still, only consumed sensors are
attached (semantic and bird's-eye views are optional), lighting is a lookup
table, and the viewer repaints only changed panels. If it still stutters:
`display.show_birdseye: false`, `display.scale: 0.5` (matters most over
remote desktop, which re-encodes every changed pixel), or `--fps 20`.

## Import ordering (do not reorder)

On Linux, habitat-sim and pygame fight over the GL context on the same X
display. So the Engine (habitat, offscreen EGL) is constructed **first**, the
viewer (SDL software rendering) is imported **lazily** after that, and this
file never imports pygame directly. Both workarounds are no-ops on macOS.
