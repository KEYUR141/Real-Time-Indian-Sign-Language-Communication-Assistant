# Real-Time Indian Sign Language Communication Assistant

## Project Work Plan

## 1. Project Goal

Build a software-only assistant that captures an Indian Sign Language (ISL) user through a webcam, recognizes a selected set of isolated signs, and presents the result as text with optional speech output.

The first release is an assistive prototype, not a replacement for a qualified interpreter. It will begin with a small controlled vocabulary and expand only after the complete pipeline works reliably.

## 2. Planned System Flow

```text
ISL user
   -> Webcam capture
   -> MediaPipe hand and pose landmarks
   -> Landmark preprocessing and sequence buffer
   -> Trained sign-recognition model
   -> Confidence and stability checks
   -> Recognized text
   -> Optional text-to-speech
```

## 3. Working Principles

1. Build a three-sign vertical prototype before collecting a large dataset.
2. Keep raw recordings, processed landmarks, and model outputs separate.
3. Split training and testing data by signer, not by random video.
4. Preserve an `unknown` result instead of forcing a prediction.
5. Record experiments, dataset versions, and decisions.
6. Obtain informed consent before recording participants.
7. Verify signs with an authoritative ISL source or a qualified ISL user.

## 4. Project Stages

### Stage 1: Domain and Requirement Study

**Estimated duration:** 3 to 5 working days

**Purpose:** Understand ISL, define the exact problem, examine available resources, identify users, and establish a realistic first-release scope.

**Activities:**

- Study ISL as a visual language.
- Distinguish sign recognition from sign-language translation.
- Review isolated and continuous recognition approaches.
- Identify users, use cases, constraints, and success criteria.
- Review dataset candidates and authoritative sign references.
- Establish consent, privacy, accessibility, and responsible-use rules.

**Exit criteria:** The scope, requirements, vocabulary-selection rules, data strategy, risks, and first architectural decisions are documented and approved by the team.

### Stage 2: Initial Vocabulary Selection

**Estimated duration:** 2 to 3 working days

**Purpose:** Select a small and verifiable set of signs for the first prototype.

**Activities:**

- Select three visually distinct signs for the pilot.
- Verify sign meaning and execution using ISLRTC or an ISL expert.
- Define label names, variants, start position, end position, and recording instructions.
- Avoid signs that cannot be distinguished using the planned features.

**Exit criteria:** A reviewed three-sign vocabulary specification is available.

### Stage 3: MediaPipe Learning Prototype

**Estimated duration:** 4 to 7 working days

**Purpose:** Understand webcam capture and landmark extraction before designing the dataset.

**Activities:**

- Capture webcam frames with Python and OpenCV.
- Detect up to two hands with MediaPipe Hand Landmarker.
- Detect selected upper-body points with Pose Landmarker.
- Display landmark connections and coordinate values.
- Test missing-hand handling, handedness, frame timestamps, and processing speed.

**Exit criteria:** A local webcam program displays stable hand and pose landmarks and records basic performance measurements.

### Stage 4: Data-Collection Application

**Estimated duration:** 5 to 8 working days

**Purpose:** Produce consistent, labelled, traceable recordings.

**Activities:**

- Create participant and recording-session forms.
- Add sign selection, countdown, recording, preview, accept, and reject controls.
- Save raw video, landmarks, and metadata.
- Validate filenames, labels, durations, and missing landmarks.
- Keep a collection log and consent record.

**Exit criteria:** The application can collect and validate a complete recording session.

### Stage 5: Pilot Dataset

**Estimated duration:** 3 to 5 working days

**Initial target:** 3 signs x 3 signers x 20 samples = 180 recordings.

**Activities:**

- Record participants under a documented protocol.
- Review samples for incorrect signs, interruptions, framing issues, and poor visibility.
- Inspect class and signer balance.
- Confirm that landmarks can be extracted consistently.

**Exit criteria:** A reviewed, versioned pilot dataset is ready for preprocessing.

### Stage 6: Preprocessing Pipeline

**Estimated duration:** 4 to 7 working days

**Purpose:** Convert variable-length recordings into reliable model inputs.

**Activities:**

- Extract hand and selected pose landmarks.
- Normalize position, scale, and orientation where appropriate.
- Preserve landmark visibility or missing-value masks.
- Resample sequences to a fixed length or use masking.
- Encode labels and create signer-independent data splits.
- Generate dataset quality reports.

**Exit criteria:** A reproducible command converts a dataset version into train, validation, and test features.

### Stage 7: Baseline Model

**Estimated duration:** 4 to 7 working days

**Purpose:** Establish a simple, measurable reference before using a complex sequence model.

**Activities:**

- Train a simple classifier on static or summarized landmarks.
- Record model configuration and random seeds.
- Measure accuracy, macro F1, per-class recall, and confusion matrix.
- Evaluate using an unseen signer.
- Document errors and data-quality issues.

**Exit criteria:** A reproducible baseline and evaluation report are available.

### Stage 8: Dynamic Sign Model

**Estimated duration:** 1 to 2 weeks

**Purpose:** Learn hand and body movement across time.

**Activities:**

- Train an LSTM, GRU, temporal convolution, or small Transformer model.
- Compare models using the same signer-independent split.
- Tune sequence length, selected landmarks, dropout, and learning rate.
- Select the model using validation results only.
- Export the best model with its label map and preprocessing configuration.

**Exit criteria:** The selected sequence model outperforms or meaningfully complements the baseline.

### Stage 9: Real-Time Inference

**Estimated duration:** 5 to 8 working days

**Purpose:** Convert live landmark sequences into stable predictions.

**Activities:**

- Maintain a rolling sequence buffer.
- Run inference at a controlled interval.
- Apply confidence thresholds and temporal smoothing.
- Suppress duplicate predictions.
- Return `unknown` when evidence is insufficient.
- Measure end-to-end latency and frame rate.

**Exit criteria:** The prototype recognizes selected signs live without uncontrolled repeated output.

### Stage 10: Frontend and Backend Integration

**Estimated duration:** 1 to 2 weeks

**Purpose:** Deliver a usable demonstration application.

**Activities:**

- Implement the Python inference service and FastAPI backend.
- Build a simple accessible web interface.
- Use WebSocket communication for live session events and predictions.
- Add start, stop, correction, clear, and status controls.
- Add optional browser text-to-speech.
- Keep the original recognition output distinguishable from any corrected sentence.

**Exit criteria:** A user can complete a live recognition session through the interface.

### Stage 11: Expanded Dataset, Testing, and Demonstration

**Estimated duration:** 2 to 4 weeks

**Purpose:** Expand the vocabulary and prepare a defensible final project.

**Activities:**

- Expand toward 20 to 30 isolated signs.
- Add signers and controlled environmental variation.
- Retrain and evaluate on unseen signers.
- Test accuracy, latency, reliability, usability, and accessibility.
- Document known limitations and failure cases.
- Package the application and prepare the final demonstration.

**Exit criteria:** The final build, model, evaluation report, documentation, and demonstration are reproducible.

## 5. Indicative Timeline

| Stage | Estimated duration | Dependency |
|---|---:|---|
| 1. Domain and requirements | 3 to 5 days | None |
| 2. Vocabulary selection | 2 to 3 days | Stage 1 |
| 3. MediaPipe prototype | 4 to 7 days | Stage 1 |
| 4. Collection application | 5 to 8 days | Stages 2 and 3 |
| 5. Pilot dataset | 3 to 5 days | Stage 4 |
| 6. Preprocessing | 4 to 7 days | Stage 5 |
| 7. Baseline model | 4 to 7 days | Stage 6 |
| 8. Dynamic model | 1 to 2 weeks | Stage 7 |
| 9. Real-time inference | 5 to 8 days | Stage 8 |
| 10. Application integration | 1 to 2 weeks | Stage 9 |
| 11. Expansion and testing | 2 to 4 weeks | Stage 10 |

Stages 2 and 3 can partly overlap. The complete project is likely to require approximately 10 to 16 weeks for a small college team, depending on participant availability, vocabulary size, and model performance.

## 6. Current Status

- [x] Project direction selected
- [x] Initial architecture planned
- [x] Presentation content prepared
- [ ] Stage 1 reviewed and approved
- [ ] Three-sign vocabulary selected
- [ ] MediaPipe prototype implemented
- [ ] Pilot data collected
- [ ] Baseline model evaluated
- [ ] Dynamic model evaluated
- [ ] Real-time application integrated
- [ ] Final testing completed
