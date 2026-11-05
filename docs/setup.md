# PhysicalAI26 HW1

This repository contains the HW1 capture simulator and a reconstruction and
experiment workflow. Python dependencies are managed by [Pixi](https://pixi.sh/)
using the checked-in `pixi.lock` file.

## Setup

1. Install Pixi using the instructions at [pixi.sh](https://pixi.sh/).
2. From this directory, create the `habitat` environment:

   ```bash
   pixi install -e habitat
   ```

   The environment provides Python 3.9, Habitat-Sim 0.3.3 with Bullet,
   Habitat-Lab, and the project dependencies. The first install can take a few
   minutes.
3. Check that Habitat-Sim and Habitat-Lab import correctly:

   ```bash
   pixi run -e habitat smoke
   ```

4. Download and extract the Replica apartment scene used by the default
   configuration:

   ```bash
   pixi run -e habitat fetch-replica
   ```

   This creates `replica_v1/apartment_0/`. Keep the scene at that path for the
   default `configs/second_floor.yaml` configuration.

## Capture data

Start the interactive Habitat-Sim collector:

```bash
pixi run -e habitat python load.py
```

The default configuration is `configs/second_floor.yaml`. Captures are written
under `eval/_data/second_floor/` and include RGB and depth PNGs, camera
intrinsics, and a ground-truth pose trajectory. Use `--config` to select a
different configuration or `--output-root` to change the output directory. See
[`docs/load.md`](load.md) for controls, replay mode, and capture details.

## Reconstruction and experiments

Run commands from this repository directory with `pixi run -e habitat`. The
experiment workflow starts by declaring an experiment over a capture, then
reconstructing it:

```bash
pixi run -e habitat python api.py declare \
  --name <experiment-name> \
  --data-dir <capture-directory> \
  --floor 2

pixi run -e habitat python reconstruct.py \
  --experiment experiments/<experiment-name> \
  --no-vis

pixi run -e habitat python api.py explore experiments/<experiment-name>
```

`declare` creates an experiment directory, the first reconstruction measures
pipeline factors and records a baseline, and `explore` displays the results and
writes interactive plots. For the full workflow, SPARQL frame selection, and
query examples, see [`docs/e2e-workflow.md`](e2e-workflow.md).

## More documentation

- [`docs/pixi.md`](pixi.md): environment details and Pixi commands
- [`docs/load.md`](load.md): capture settings, controls, and outputs
- [`docs/reconstruct.md`](reconstruct.md): reconstruction options
- [`docs/api.md`](api.md): experiment and inspection commands
- [`docs/e2e-workflow.md`](e2e-workflow.md): end-to-end walkthrough
