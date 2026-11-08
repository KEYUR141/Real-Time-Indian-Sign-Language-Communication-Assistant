# Project Implementation and Development Report

**Name:** Keyur Sherke

**GitHub:** [@KEYUR141](https://github.com/KEYUR141)

**Contribution area:** Project implementation, model experimentation, application integration, and repository maintenance

**Project:** Real-Time Indian Sign Language Communication Assistant

## System Design and Implementation

I led the implementation of the isolated-sign recognition pipeline, from
dataset preparation and landmark extraction to model inference and the
application interface. The system accepts an uploaded video or a manually
bounded webcam sequence and returns a predicted sign label with supporting
confidence and quality information.

The implementation uses Google MediaPipe for landmark detection, NumPy for
feature storage and processing, scikit-learn for the SVM baseline, PyTorch for
the temporal models, FastAPI for the backend, and React with Vite for the
frontend.

## Dataset Preparation and Landmark Processing

I developed the manifest-driven workflow for INCLUDE-50 and the larger local
INCLUDE benchmark. This included resolving video paths, handling nested
`Extra` directories, preserving class and split metadata, and excluding the
unavailable `Second (Number)` class from the larger experiment.

The extraction pipeline produces compressed `.npz` archives containing
coordinates, confidence values, detection masks, timestamps, and source
metadata. It supports resuming completed outputs and reporting extraction
failures. Folder validation and manifest-aware quality reports check feature
integrity, missing files, split coverage, and class coverage.

Each temporal input contains up to 64 frames with 51 landmarks per frame:
21 landmarks for each hand and nine selected upper-body pose landmarks.
Coordinates, confidence values, and detection indicators form 255 features
per frame. Shoulder-based normalization reduces differences in position and
scale, while padding masks and true sequence lengths distinguish real frames
from padded timesteps.

## Model Development and Experiments

I implemented and ran the baseline model experiments, beginning with
INCLUDE-50 and extending to the 262-class local INCLUDE benchmark.

- **SVM — Support Vector Machine:** Converts each sequence into a
  1,377-dimensional summary vector and classifies it using an RBF kernel.
- **GRU — Gated Recurrent Unit:** An additional sequence-model comparison
  evaluated on INCLUDE-50 validation data.
- **LSTM — Long Short-Term Memory:** Processes frames sequentially and
  classifies the resulting temporal representation.
- **Compact Transformer:** Uses projected landmark features, learned positional
  embeddings, a classification token, and masked self-attention to classify
  the gesture sequence.

Validation Macro F1 guided configuration and epoch selection. Final benchmark
runs refitted the selected configurations on training plus validation data
before evaluating the held-out test split. Experiments also examined different
random seeds and a larger Transformer configuration.

### Recorded Full Local INCLUDE Results

The benchmark contains 4,284 videos across 262 classes, including 816 test
videos. These are dataset benchmark results, not measured live-user accuracy.

| Model | Test accuracy | Test Macro F1 | Correct test videos |
|---|---:|---:|---:|
| RBF SVM | 53.92% | 51.05% | 440/816 |
| LSTM | 64.83% | 64.01% | 529/816 |
| Compact Transformer | 94.36% | 94.47% | 770/816 |

The Compact Transformer was selected for application integration. Its final
full-INCLUDE configuration used seed 42 and a validation-selected epoch
count of 62.

## Inference and Application Integration

I implemented checkpoint loading and landmark-sequence prediction so that
the trained model can be reused without retraining during inference. The
prediction workflow restores the saved architecture, normalization statistics,
and label mapping.

Application work included:

- Single-video prediction with ranked labels and landmark coverage.
- Desktop webcam capture with explicit start/stop boundaries for one gesture.
- Quality and confidence checks that can return `unknown`.
- Evaluation logging and helpers for temporal stability and duplicate suppression.
- FastAPI endpoints for prediction, status, reference videos, and model assets.
- A React interface with reference-video playback, landmark visualization,
  webcam capture, video upload, and prediction feedback.
- Browser landmark submission to the backend for classification.

## Testing and Observed Limitations

The implementation includes automated tests for manifest handling, sampling,
normalization, feature loading, quality reports, model behavior, inference,
and API functionality. Experiment outputs include metric reports and confusion
matrices for examining class-level performance.

Recorded INCLUDE clips produced strong predictions, but new webcam recordings
sometimes returned `unknown` or an incorrect label. The project documentation
records this gap rather than treating the dataset score as proof of reliable
recognition for every new signer. Capture boundaries, occlusion, landmark
quality, and recording differences remain important investigation areas.

## Documentation and Repository Maintenance

I maintained the implementation repository and prepared documentation covering
the development stages, model comparisons, training terminology, evaluation
metrics, inference workflow, and remaining limitations.

The repository also includes a PowerShell command reference, presentation and
viva preparation notes, and a README with the team, application screenshots,
benchmark results, and project showcase photograph. Project report and
presentation preparation accompanied the implementation work.

## Outcome and Next Scope

The completed work provides an end-to-end prototype for recognizing isolated
ISL signs within the evaluated vocabulary. The main next steps are collecting
consented recordings from more users, testing across signers and environments,
improving consistency between training and live capture, and calibrating
rejection using development data.

## Supporting Project Records

- [Full INCLUDE benchmark](../documents/stage%206/03-full-include-model-benchmark.md)
- [INCLUDE-50 benchmark](../documents/stage%206/01-include50-model-benchmark.md)
- [Model workflows and terminology](../documents/stage%206/02-model-workflows-and-training-terminology.md)
- [Command reference](../cmd.txt)
- [Presentation preparation](../presentation.md)
- [Project README](../README.md)
