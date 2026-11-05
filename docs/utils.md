# `utils.py` — reconstruction pipeline

> **TA-internal implementation documentation.** This repository contains the
> complete homework implementation and reference materials.

`utils.py` implements RGB-D unprojection, point-cloud preparation, pairwise
registration, trajectory composition, and trajectory scoring. `reconstruct.py`
is the CLI; `api.py` supplies the experiment and RDF workflow.

## Camera and depth input

`depth_image_to_point_cloud(rgb, depth_m, width, height, hfov)` expects BGR
`uint8` RGB data, a same-sized depth array already converted to metres, and
the capture's camera intrinsics. It uses a pinhole camera with square pixels,
an image-centred principal point, and focal length derived from width and
horizontal field of view. Pixels with depth at or below zero are dropped.
The returned cloud uses metres and camera axes (+X right, +Y down, +Z forward).

The collector stores depth PNGs as uint16 millimetres. `reconstruct` converts
them to metres and reads `intrinsics.json` from the capture directory. The RGB
and depth image dimensions must match the recorded intrinsics.

## Pipeline stages

| Function | Role |
|---|---|
| `preprocess_point_cloud` | Voxel-downsamples the cloud, estimates normals, and computes FPFH descriptors. |
| `local_icp_algorithm` | Refines an initial transform with Open3D point-to-plane ICP. |
| `my_local_icp_algorithm` | Custom point-to-point ICP backend selected by `--version my_icp`. |
| `reconstruct` | Builds each cloud, registers it to the previous consumed cloud, composes camera poses, and optionally merges a map. Reports pipeline-factor measurements through its callback. |
| `remove_ceiling_points` | Crops the merged map after trajectory construction; it does not change the poses. |
| `visualize_and_evaluate` | Scores camera-centre Mean L2 against `GT_pose.npy`; optionally opens an Open3D view. |

For each pair, `reconstruct` obtains a coarse transform using mutual FPFH
feature matching and Open3D RANSAC with edge-length and distance checks. It
then refines that transform using either Open3D point-to-plane ICP (`open3d`)
or the custom point-to-point ICP (`my_icp`). The default voxel size is 5 cm;
the ICP correspondence gate is 1.5 times the voxel size. The RANSAC seed is
provided by the CLI.

The callback reports the eight first-order pipeline factors from state already
available during this pass: two point counts per image and six
`RegistrationResult` reads for each adjacent pair. See [`factors.md`](factors.md)
for their definitions.

## `reconstruct` contract

- **Input.** A capture directory with `rgb/`, `depth/`, and
  `intrinsics.json`. Trajectory scoring also reads `GT_pose.npy` when present.
  With `frames=None`, the implementation sorts RGB and depth filenames
  separately and pairs them by list position up to the shorter list. With a
  frame list, it loads the named stems in the supplied order.
- **Output.** `(global_pcd, pred_cam_pos)`: an optional ceiling-cropped map in
  the first camera frame and camera centres composed from the pairwise
  transforms. `build_cloud=False` skips map merging but still builds the
  trajectory. Ground-truth poses are not used for registration or alignment.
- **Selection gaps.** `reconstruct.py` sorts query-selected stems and records
  contiguous segments for splice metadata, then passes the flattened frame
  list to this function. Each selected frame is registered to the previous
  selected frame, including across segment boundaries. The segments are not
  reconstructed independently.
- **Scoring.** `visualize_and_evaluate` matches requested frame stems to their
  ordinal positions in the capture, reconciles the predicted and GT axes into
  the first camera frame, and reports mean Euclidean camera-centre error. It
  does not fit an alignment to the GT. Missing or empty GT yields infinite
  Mean L2.
