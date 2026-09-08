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

### Stage 2: Data-Collection Application and Pilot Setup

**Estimated duration:** 1 to 2 weeks

**Purpose:** Start implementation early by building a reliable application for recording labelled ISL videos and their metadata.

**Activities:**

- Select and verify three visually distinct ISL signs for the pilot.
- Define label names, accepted variants, start position, end position, and recording instructions.
- Create participant, consent, vocabulary, and recording-session records.
- Add sign selection, countdown, recording, preview, accept, reject, and re-record controls.
- Save raw video and metadata using a consistent directory and filename scheme.
- Validate labels, participant codes, video duration, and required metadata.
- Keep landmark extraction optional until the MediaPipe prototype is validated in Stage 3.

**Exit criteria:** The application can complete a controlled three-sign recording session and preserve accepted raw videos with valid metadata.

### Stage 3: MediaPipe Learning and Integration Prototype

**Estimated duration:** 4 to 7 working days

**Purpose:** Validate landmark extraction and integrate it with recordings before beginning pilot data collection.

**Activities:**

- Capture webcam frames with Python and OpenCV.
- Detect up to two hands with MediaPipe Hand Landmarker.
- Detect selected upper-body points with Pose Landmarker.
- Display landmark connections and coordinate values.
- Test missing-hand handling, handedness, frame timestamps, and processing speed.
- Run landmark extraction on videos produced by the Stage 2 application.
- Decide whether landmarks are saved during recording, extracted offline, or both.

**Exit criteria:** MediaPipe produces stable hand and pose landmarks from both the live webcam and videos saved by the collection application.

### Stage 4: Pilot Dataset

**Estimated duration:** 3 to 5 working days

**Initial target:** 3 signs x 3 signers x 20 samples = 180 recordings.

**Activities:**

- Record participants under a documented protocol.
- Review samples for incorrect signs, interruptions, framing issues, and poor visibility.
- Inspect class and signer balance.
- Confirm that landmarks can be extracted consistently.

**Exit criteria:** A reviewed, versioned pilot dataset is ready for preprocessing.

### Stage 5: Preprocessing Pipeline

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

### Stage 6: Baseline Model

**Estimated duration:** 4 to 7 working days

**Purpose:** Establish a simple, measurable reference before using a complex sequence model.

**Activities:**

- Train a simple classifier on static or summarized landmarks.
- Record model configuration and random seeds.
- Measure accuracy, macro F1, per-class recall, and confusion matrix.
- Evaluate using an unseen signer.
- Document errors and data-quality issues.

**Exit criteria:** A reproducible baseline and evaluation report are available.

### Stage 7: Dynamic Sign Model

**Estimated duration:** 1 to 2 weeks

**Purpose:** Learn hand and body movement across time.

**Activities:**

- Train an LSTM, GRU, temporal convolution, or small Transformer model.
- Compare models using the same signer-independent split.
- Tune sequence length, selected landmarks, dropout, and learning rate.
- Select the model using validation results only.
- Export the best model with its label map and preprocessing configuration.

**Exit criteria:** The selected sequence model outperforms or meaningfully complements the baseline.

### Stage 8: Real-Time Inference

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

### Stage 9: Frontend and Backend Integration

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

### Stage 10: Expanded Dataset, Testing, and Demonstration

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
| 2. Collection application and pilot setup | 1 to 2 weeks | Stage 1 |
| 3. MediaPipe prototype and integration | 4 to 7 days | Stage 1; may overlap Stage 2 |
| 4. Pilot dataset | 3 to 5 days | Stages 2 and 3 |
| 5. Preprocessing | 4 to 7 days | Stage 4 |
| 6. Baseline model | 4 to 7 days | Stage 5 |
| 7. Dynamic model | 1 to 2 weeks | Stage 6 |
| 8. Real-time inference | 5 to 8 days | Stage 7 |
| 9. Application integration | 1 to 2 weeks | Stage 8 |
| 10. Expansion and testing | 2 to 4 weeks | Stage 9 |

Stages 2 and 3 should overlap where possible. The collection interface and raw-video storage can be developed while MediaPipe is being learned. Pilot collection starts only after both stages pass their exit criteria. The complete project is likely to require approximately 10 to 16 weeks for a small college team, depending on participant availability, vocabulary size, and model performance.

## 6. Current Status

- [x] Project direction selected
- [x] Initial architecture planned
- [x] Presentation content prepared
- [ ] Stage 1 reviewed and approved
- [ ] Stage 2 collection application and three-sign setup completed
- [ ] MediaPipe prototype implemented
- [ ] Pilot data collected
- [ ] Baseline model evaluated
- [ ] Dynamic model evaluated
- [ ] Real-time application integrated
- [ ] Final testing completed
