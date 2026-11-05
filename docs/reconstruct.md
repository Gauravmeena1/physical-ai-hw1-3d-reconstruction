# `reconstruct.py` — reconstruction CLI

> **TA-internal implementation documentation.** This repository contains the
> complete homework implementation and reference materials.

`reconstruct.py` orchestrates the pipeline in `utils.py` and can record run
results in an assessed experiment Turtle file. The default path uses Open3D
FPFH/RANSAC followed by Open3D ICP. `my_icp` selects the custom point-to-point
ICP implementation.

## Assessed experiment workflow

Run commands from the repository root in the Habitat Pixi environment:

```bash
pixi run -e habitat python api.py declare --name <exp> --data-dir <capture>
pixi run -e habitat python reconstruct.py --experiment experiments/<exp> --no-vis
```

`declare` creates `experiments/<exp>/` with `<exp>.ttl`, `plots/`, and `queries/`;
then `reconstruct.py --experiment experiments/<exp>` acts as the first-pass measuring run: it measures all pipeline factors inline, seals the
machine section, writes the baseline run, and saves the reconstruction visualization as `reconstruction.png` in the experiment directory. Subsequent runs with `--selection-query`
perform filtered reconstructions on the sealed file and refresh `reconstruction.png`. By default, subsequent runs are dry-run (`no-write`)
and calculate/print the result without changing the experiment. Use `--override-experiment` to write the run node.
`api.py explore experiments/<exp>` writes `dashboard.html` plus three individual
interactive plot pages under `experiments/<exp>/plots/`. Add
`--launch-dashboard` to serve the Panel dashboard on localhost after saving.
The reusable-query directory is `experiments/<exp>/queries/`.

## Frame selection

Without a query, the CLI reconstructs all common RGB/depth PNG stems in numeric
order. With `--selection-query` and `--experiment`, it
runs a local SPARQL `SELECT` over that experiment file. The query must bind
`?frame` (a frame IRI) or `?frameIndex` (an integer). Returned stems are
validated against the experiment's frames, sorted, and split into maximal
contiguous numeric segments before being flattened into one selected frame
list. Reconstruction registers each frame to the previous selected frame,
including across segment boundaries; splice metadata records these boundaries,
but the segments are not reconstructed independently. This selection is defined
by the query, including any value bands in its `FILTER` clauses.

```bash
pixi run -e habitat python reconstruct.py \
  --experiment experiments/<exp> --selection-query experiments/<exp>/queries/frames.rq --no-vis
```

An empty query result produces no reconstruction and writes no run.

## Flags

| Flag | Meaning |
|---|---|
| `--experiment` | Experiment directory such as `experiments/<name>` (required); names the capture via `hw1:batchFile` / sealed `hw1:batchPath` — no capture flag exists. An unassessed declaration triggers the measuring first pass. Its recorded `icpBackend`, when present, selects the backend and cannot be overridden with a different `--version`. |
| `--version` | `open3d` or `my_icp`; defaults to the experiment's `icpBackend`, or `open3d` without one. |
| `--voxel-size` / `--seed` | Registration voxel size (default `0.05` m) and Open3D RANSAC seed (default `0`). |
| `--selection-query` | SPARQL file selecting frames over the `--experiment` graph. |
| `--no-vis` | Skip the Open3D visualization window. |
| `--override-experiment`, `--write` | Write reconstruction evidence to the experiment file (default is no-write). |
| `--no-write` | Explicit dry run (default behavior): print the metric without writing to the experiment. |

`--experiment` is required: the capture always comes from the experiment's declared `hw1:batchFile` (measuring run) or sealed `hw1:batchPath` (later runs). See [`utils.md`](utils.md) for the
geometry and metric contracts, [`api.md`](api.md) for experiment records,
and [`e2e-workflow.md`](e2e-workflow.md) for the end-to-end walkthrough.
