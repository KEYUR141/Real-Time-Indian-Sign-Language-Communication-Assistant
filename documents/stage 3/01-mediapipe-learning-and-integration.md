# Stage 3: MediaPipe Learning and Integration

## 1. Purpose

Stage 3 established a reusable video-to-landmark pipeline before model
training. The objective was to verify that MediaPipe could consistently
represent isolated INCLUDE signs using hand and upper-body motion, while
preserving enough provenance to reproduce and audit every generated feature.

## 2. Selected Landmark Representation

Each sampled frame contains 51 landmarks:

- 21 left-hand landmarks;
- 21 right-hand landmarks;
- 9 pose landmarks covering the upper body;
- 3 coordinates per landmark;
- one confidence value and one detection indicator per landmark.

Face landmarks are disabled for this benchmark. A temporal model receives 255
values per frame: 153 coordinates, 51 confidence values, and 51 detection
indicators. Sequences are sampled or padded to 64 frames.

## 3. Implemented Components

The implementation separates responsibilities so that each part can be tested:

| Component | Responsibility |
|---|---|
| `mediapipe_detector.py` | Run hand and pose MediaPipe Tasks and map results into the fixed schema |
| `feature_schema.py` | Define landmark ordering, slices, and schema version |
| `temporal_sampling.py` | Select up to 64 frames and create padding masks |
| `normalization.py` | Normalize position and scale while respecting missing landmarks |
| `extractor.py` | Decode videos, reuse a detector, assign timestamps, and save arrays |
| `visualization.py` | Draw and inspect detected landmarks |

The scripts `inspect_mediapipe_video.py` and
`visualize_mediapipe_video.py` support manual inspection. The manifest-driven
entry point is `extract_landmark_folder.py`.

## 4. Detector and Timestamp Decisions

MediaPipe video mode requires strictly increasing timestamps. Creating a new
detector for every video is expensive, so one detector is reused across the
run. `MediaPipeTimestampClock` maps each video's relative timestamps onto one
monotonically increasing detector timeline. The original video-relative
timestamps are still stored in the archive.

Hand slots are assigned from MediaPipe handedness rather than detection order.
If two detections claim the same side, the higher-confidence detection is
retained. A manual handedness-swap option is available for mirrored-input
investigation, but is not enabled by default.

Missing detections remain zero-filled and are accompanied by explicit masks.
They are not silently interpreted as observed coordinates.

## 5. Normalization and Sampling

Long videos are sampled uniformly over their full duration. Short videos keep
all real frames and are right-padded. Archives retain:

- the valid-timestep mask;
- source frame indices;
- source-relative timestamps;
- source frame count and frames per second;
- raw and normalized coordinates;
- confidence and detection arrays;
- normalization validity.

Normalization uses visible pose or hand geometry when available and records
when a timestep cannot be normalized reliably.

## 6. Archive and Provenance Contract

Each `.npz` archive includes the feature arrays plus its text label, split,
manifest-relative video path, resolved source path, and schema version. Output
paths include the official split, canonical class label, and a stable ID made
from the complete manifest path. This prevents collisions between repeated
video basenames in different folders.

The text label and manifest path are the stable identity. Numeric label IDs
are benchmark-specific: INCLUDE-50 and full INCLUDE use different label maps.
Model loaders therefore derive the active numeric ID from the selected
manifest and label map, allowing one landmark archive to be reused safely.

## 7. Verification

Automated tests cover:

- hand assignment and optional handedness swapping;
- missing and duplicate-side detections;
- the selected pose points;
- timestamp monotonicity across reused detector calls;
- fixed-length temporal sampling and padding;
- landmark normalization;
- collision-free output paths;
- atomic archive writing and required metadata;
- `.MOV` to `.MP4` local path resolution;
- nested `Extra` manifest folders.

Manual pilot extraction and validation also confirmed that MediaPipe features
could be produced and reopened successfully.

## 8. Stage Decision

Stage 3 is complete for offline INCLUDE processing. The project selected
offline extraction because it keeps raw videos immutable, makes preprocessing
reproducible, and allows the same features to be used by several classifiers.
Live webcam integration remains a later real-time inference task.

