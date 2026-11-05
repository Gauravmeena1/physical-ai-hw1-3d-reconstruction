# `api.py` — local experiment and data inspection CLI

> **TA-internal implementation documentation.** Keep this checkout and its
> complete homework implementation within the course staff team.

`api.py` declares experiments, records pipeline-derived factor occurrences and
run outcomes, and provides inspection and SPARQL queries over local Turtle
files. `explore` saves plot artifacts for assessed experiments; it does not
change their Turtle. Reconstruction and trajectory scoring are performed by
`reconstruct.py`.

## Workflow

1. `declare` scaffolds an experiment with its batch, reconstruction settings,
   and prediction notes. Factors are measured automatically; the declaration
   does not select them.
2. The first `reconstruct.py --experiment experiments/<name>` pass measures the eight
   first-order pipeline factors from pipeline state, writes their occurrence records, and
   seals the machine section with `hw1:declarationDigest`.
3. `explore` displays summaries and saves interactive plots for assessed experiments; `query` executes local SPARQL.
4. Later reconstruction passes can use `--selection-query` and append run
   records without changing the sealed factor evidence.

## Commands

| Command | What it does |
|---|---|
| `declare` | Create `experiments/<name>/` with `<name>.ttl`, `plots/`, and `queries/`, then scaffold the declaration. |
| `explore` | Display captures, declarations, assessed experiments, and comparisons; assessed experiments also save four Panel HTML artifacts under `plots/`. |
| `query` | Run a read-only SPARQL query against a local Turtle file. |
| `batch2ttl` | Write shared batch and frame structure. |

Use `--help` for current arguments.

Pass an experiment directory (for example `experiments/second_floor`) to
`explore`, `query`, and `reconstruct.py --experiment`. Run
`api.py explore experiments/<name> --launch-dashboard` to save the plots and
serve their dashboard on localhost. See [`plot.md`](plot.md) for the three
views, artifact names, and recorded-data limits.

## Data model

Each measured result is an experiment-scoped `hw1:Factor` occurrence linked
to a reusable `hw1:QualityFactor` definition. `hw1:SingleImageFactor`
occurrences target a depth image; `hw1:DepthPairFactor` occurrences identify
ordered adjacent depth images through `hw1:hasPrevious` and
`hw1:hasCurrentFrame`. Occurrences store `hw1:value` and `hw1:evaluationState`.

`ReconstructionRun` stores `mapMeanL2` directly. `icpBackend` has the
`ReconstructionSetting` role. Experiments on the same capture share batch,
frame, and image IRIs while their factor occurrences, settings, and runs
remain experiment-scoped.

The ontology maps each reusable factor definition to its corresponding
observable property. This mapping lets the code interpret the generic
occurrence records while retaining data-driven factor metadata.

## Factor support

| Pipeline stage | Factor definitions |
|---|---|
| Back-projected and downsampled depth cloud | `RawPointCount`, `VoxelizedPointYield` |
| Adjacent depth image registration | `CoarseRANSACFitness`, `CoarseRANSACRMSE`, `CoarseRANSACCorrespondences`, `ICPInlierFitness`, `ICPInlierRMSE`, `ICPInlierCorrespondences` |

See [`factors.md`](factors.md) for measurements, and
[`triplestore.md`](triplestore.md) for the graph and file lifecycle.
