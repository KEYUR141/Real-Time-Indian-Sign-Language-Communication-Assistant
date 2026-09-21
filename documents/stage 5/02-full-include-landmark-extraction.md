# Full INCLUDE Landmark Extraction Plan

## Scope

The local full INCLUDE benchmark excludes the unavailable class
`Second (Number)`. Its eight official manifest videos are absent locally. The
resulting benchmark contains 262 classes and 4,284 available videos.

The existing 958 INCLUDE-50 archives are reused. The next MediaPipe run should
therefore convert only the remaining 3,326 videos.

## Configuration

`configs/landmarks_include.toml` selects:

- the full `include` manifests;
- the full INCLUDE label map;
- the existing raw-video and processed-feature roots;
- the unchanged 64-frame, 51-landmark schema;
- explicit exclusion of `Second (Number)`.

Numeric label IDs differ between INCLUDE-50 and full INCLUDE. Feature archives
are reusable because their stable identity is the text label plus manifest
path. Quality and model loaders derive numeric IDs from the active benchmark's
label map.

## Safe Plan Check

This command resolves manifest paths and prints counts only. It does not load
MediaPipe models, decode frames, or create `.npz` files:

```powershell
uv run python scripts/extract_landmark_folder.py `
  --config configs/landmarks_include.toml `
  --reuse-subset include50 `
  --plan-only
```

Expected selection:

```text
Benchmark videos: 4284
Classes selected: 262
Reusable landmark archives: 958
MediaPipe conversions required: 3326
```

The official INCLUDE-50 and full-INCLUDE manifests assign some shared videos to
different splits. Reuse therefore matches the stable relative video path and
creates the full-INCLUDE split identity; it does not simply skip matching
split-specific output paths.

## Extraction Command

Run this only when ready to perform the long MediaPipe conversion:

```powershell
uv run python scripts/extract_landmark_folder.py `
  --config configs/landmarks_include.toml `
  --reuse-subset include50 `
  --report-name full_include_extraction_report.json
```

Do not add `--overwrite`. A complete matching output is resumable, while a
missing or structurally incomplete output is regenerated. The report is
checkpointed after each sample, so an interrupted run retains its progress.

## Post-extraction Audit

The quality report must evaluate the complete 4,284-feature benchmark, not only
the incremental 3,326 files:

```powershell
uv run python scripts/report_landmark_dataset_quality.py `
  --config configs/landmarks_include.toml `
  --report-name full_include_quality_report.json
```

Expected manifest coverage after successful extraction:

```text
Expected Manifest Samples: 4284
Valid Manifest Features: 4284
Missing Manifest Features: 0
Invalid Manifest Features: 0
```

The 21 earlier pilot archives may still appear as unexpected features. They are
not selected by the manifests and are not used for training.
