# Initial Technical Decisions

## 1. Architectural Baseline

```text
Webcam
  -> Frame acquisition
  -> MediaPipe hand and pose extraction
  -> Feature normalization
  -> Temporal sequence buffer
  -> Team-trained classifier
  -> Confidence and stability policy
  -> Text result
  -> Optional browser speech
```

## 2. MediaPipe Decision

Use MediaPipe Tasks for feature extraction rather than training an image-based hand detector.

### Initial features

- Up to 21 landmarks for each hand
- Handedness result
- Selected pose landmarks around shoulders, elbows, wrists, nose, and mouth area if needed
- Per-landmark visibility or missing-value mask where available
- Frame timestamp

### Why this is appropriate

- It reduces raw images to meaningful spatial features.
- It requires less training data than direct video modelling.
- It can run interactively on ordinary hardware.
- It lets the team inspect and debug model inputs visually.

### Limitations

- Landmarks can fail during occlusion, blur, and poor lighting.
- Coordinates do not automatically identify an ISL sign.
- Fine finger contact and facial grammar may be missed.
- Handedness may require careful handling when webcam frames are mirrored.

MediaPipe is the feature extractor. The sign-recognition classifier remains the model trained by the project team.

## 3. Feature Strategy

Raw pixel coordinates should not be passed directly to the final model without testing normalization. Candidate transformations include:

- Translation relative to a stable reference such as the wrist or shoulder centre
- Scaling relative to shoulder width, palm size, or another stable distance
- Selected joint angles and inter-landmark distances
- Motion differences between adjacent frames
- A mask distinguishing an absent landmark from a real zero value

Every transformation must be fitted or defined using training data and applied identically during inference.

## 4. Model Strategy

### Baseline

Train a simple model on one frame or sequence-summary features. Candidates include Random Forest, Support Vector Machine, and a small multilayer perceptron.

### Sequence model

Train an LSTM, GRU, temporal convolution network, or small Transformer on ordered landmark frames. Start with LSTM or GRU because the team can understand and debug it more easily.

### Evaluation

Compare models using the same signer-independent split and the same preprocessing version. Do not select a model based only on training accuracy.

## 5. Inference Policy

A raw highest-probability class is not automatically a valid output. The runtime pipeline should include:

1. Minimum landmark-quality check
2. Fixed sequence window or detected sign segment
3. Model probability or confidence check
4. Temporal agreement across predictions
5. Unknown or no-output state
6. Duplicate suppression and release condition

These controls are part of the recognition system, not optional interface polish.

## 6. Application Boundary

### Local prototype

Begin with a single Python process containing webcam capture, MediaPipe, model inference, and an OpenCV preview.

### Integrated application

After inference is stable, separate the system into:

- Browser frontend for camera controls, status, text, and speech
- FastAPI backend for session control and inference coordination
- WebSocket connection for continuous session events and prediction updates
- Model package containing preprocessing, labels, weights, and configuration

WebSocket is useful for live interaction because either side can send updates over an open connection. It is not used to obtain the training dataset.

## 7. Optional Language Model

An LLM may later propose a corrected sentence from accepted sign labels. It must be optional and visibly separated from raw recognition.

Required guardrails:

- Show the original recognized labels.
- Label reconstructed text as a suggestion.
- Do not invent missing critical information.
- Require user confirmation before speaking or sharing corrected text.
- Disable the feature for the first prototype.

## 8. Initial Technology Choices

| Area | Initial choice |
|---|---|
| Language | Python |
| Webcam and frames | OpenCV |
| Landmarks | MediaPipe Tasks |
| Data arrays | NumPy and Pandas |
| Baseline ML | scikit-learn |
| Sequence model | PyTorch or TensorFlow, to be finalized before Stage 7 |
| Backend | FastAPI |
| Live communication | WebSocket |
| Frontend | Simple web interface, framework selected during Stage 9 |
| Speech | Browser SpeechSynthesis where supported |
