# Pixi environment

The project uses a lockfile-backed Pixi environment named `habitat`. It pins
Python 3.9 for Habitat-Sim 0.3.3 and installs the reconstruction, capture,
RDF, test, and dashboard dependencies.

## Usage

```bash
pixi install -e habitat
pixi run -e habitat smoke
pixi run -e habitat python -m pytest tests/test_pipeline_workflow.py tests/test_new_factors.py tests/test_explore_dashboard.py -v
pixi shell -e habitat
```

Panel 1.4.5 and Bokeh 3.4.3 provide the experiment dashboard and self-contained
HTML plot exports. Both are pinned for the workspace's Python 3.9 runtime.

## Environment details

- `pixi.toml` uses the `conda-forge` and `aihabitat` channels and pins
  `habitat-sim` 0.3.3 with Bullet support.
- NumPy is pinned to 1.26.4 to match Habitat-Lab.
- Habitat-Lab and the local `simulator` package are installed in editable mode.
- Linux and macOS environments include their respective OpenMP runtime
  dependencies for Open3D.
