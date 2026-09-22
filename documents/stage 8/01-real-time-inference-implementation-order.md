# Stage 8: Real-Time Inference Implementation Order

## Objective

Convert the locked compact Transformer into a responsive webcam recognizer
without changing its training-time landmark schema or using the frozen test
split for threshold selection.

## Implementation Order

### 1. Checkpoint loader — implemented

Load `selected_final.pt`, validate its architecture metadata, reconstruct the
trained model, restore normalization statistics and label names, select the
requested device, and enter evaluation mode.

**Exit criterion:** A trusted project checkpoint loads on CPU or CUDA and
rejects incomplete or incompatible metadata.

### 2. Landmark-sequence predictor — implemented

Accept an in-memory `[timesteps, 255]` sequence, apply the checkpoint's
training-derived normalization to real frames, preserve zero padding, execute
in inference mode, and return ranked labels and probabilities.

**Exit criterion:** One API returns deterministic top-k predictions with
validated shapes and sequence lengths.

### 3. Unit tests — implemented

Test checkpoint reconstruction, label mapping, normalization, padding,
top-k ordering, malformed inputs, and CPU prediction. Later stages add tests
for buffering, rejection, and smoothing state.

**Exit criterion:** Inference tests pass without accessing the external
dataset or requiring a GPU.

### 4. Single-video prediction script — implemented and smoke-tested

Decode one video, run the unchanged MediaPipe extraction pipeline in memory,
construct the 255-value temporal features, load the final Transformer, and
print ranked predictions plus landmark coverage.

**Exit criterion:** A known video can be classified without creating an
intermediate `.npz` file.

### 5. Rolling webcam buffer — implemented

Maintain the most recent 64 landmark frames, track valid length and detection
coverage, reset safely, and expose immutable model-ready snapshots. Capture
can run every frame while classification runs at a configurable interval.

**Exit criterion:** Buffer behavior is deterministic for empty, filling, full,
sliding, and reset states.

### 6. Unknown rejection — implemented; calibration pending

Reject predictions when the buffer is not ready, hand coverage is inadequate,
maximum probability is low, or the margin between the two leading classes is
small. Calibrate thresholds with validation and separate live-development
recordings, never the frozen 816-video test split.

**Exit criterion:** Neutral poses, missing hands, transitions, and out-of-scope
motions return `unknown` instead of a forced INCLUDE class.

### 7. Temporal smoothing and duplicate suppression — implemented

Require repeated agreement across prediction windows, introduce an acceptance
cooldown, and require a neutral or changed state before emitting the same sign
again.

**Exit criterion:** One held sign produces one stable output rather than a
stream of duplicates.

### 8. Webcam demonstration — implemented; live calibration pending

Combine capture, MediaPipe, rolling buffer, prediction, rejection, and
smoothing. Display landmarks, buffer readiness, current candidate, confidence,
accepted text, FPS, and inference latency.

**Exit criterion:** A user can start, reset, clear, and stop a live session
while receiving stable predictions.

### 9. Latency and live-user evaluation — instrumentation implemented; trials pending

Measure MediaPipe, model, and end-to-end latency separately on CPU and CUDA.
Evaluate known signs, unknown motions, lighting, distance, backgrounds, and
different users. Record accuracy and rejection behavior without altering the
frozen dataset benchmark.

**Exit criterion:** The selected runtime device and prediction interval meet a
documented responsiveness target with known failure cases.

### 10. FastAPI and frontend integration

Wrap the tested inference engine behind a session-oriented FastAPI service and
WebSocket interface. Keep model logic independent of transport and UI code.
Add accessible controls, recognized text, correction, clearing, and optional
text-to-speech.

**Exit criterion:** The browser can complete a live recognition session while
preserving `unknown`, confidence, and original versus corrected output.

## Delivery Sequence

```text
offline inference parity
    -> live landmark buffering
    -> safe prediction acceptance
    -> measurable webcam prototype
    -> application integration
```

Each stage must pass its tests and exit criterion before the next layer is
allowed to depend on it.
