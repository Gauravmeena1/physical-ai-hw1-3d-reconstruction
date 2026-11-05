# Experiment plots

`api.py explore experiments/<name>` reads an assessed experiment, prints its
terminal summary, and writes four self-contained Panel/Bokeh HTML files under
`experiments/<name>/plots/`:

| File | View |
|---|---|
| `dashboard.html` | Tabbed dashboard containing the three views below. |
| `timelines.html` | One factor-value timeline per factor. |
| `distributions.html` | Finite recorded values by factor, with missing and non-finite counts in each title. |
| `tradeoff.html` | Recorded `mapMeanL2` against retained-frame fraction. |

Pair measurements are placed at their current frame index. Their hover detail
shows the previous frame, so the displayed point identifies the ordered pair.
Hovering over any timeline data point displays thumbnails of the frame's RGB and
depth image pair (or current and previous image pairs for pair measurements).
The timeline leaves gaps where a measurement is absent or non-finite, and the
distribution excludes non-finite values.

The tradeoff uses only reconstruction runs already recorded in the experiment.
The current writer stores one `baseline` and one `selected` run, so one
experiment contributes at most two points; `explore` does not run a sweep.
`mapMeanL2` is the only outcome currently written by the reconstruction CLI.

Add `--launch-dashboard` to save the artifacts and then serve the dashboard on
localhost. Exploring multiple assessed experiment directories saves artifacts
for each and serves them in separate tabs. Declarations and capture
directories have no measured factor values, so they produce no plots.
