# Real-Time Indian Sign Language Communication Assistant

## Project Progress Report

**Presentation status:** prepared for the current project review  
**Project scope:** isolated Indian Sign Language recognition from an ordinary RGB webcam, with recognized signs converted into text and later optional speech

## 1. Executive summary

The project has completed the dataset, landmark extraction, data-quality, model benchmarking, checkpoint loading, offline inference, and initial webcam-inference stages.

The strongest locked model is a compact Transformer trained on the locally complete portion of the full INCLUDE dataset:

- 262 sign classes
- 4,284 videos
- 3,121 training videos
- 347 validation videos
- 816 test videos
- 94.36% test accuracy
- 94.47% test Macro F1
- 770 of 816 test videos classified correctly

The model can classify a known video directly through the MediaPipe-to-Transformer pipeline. A test using the INCLUDE video for `hat` produced `hat` as the top prediction with 100% reported probability.

The current challenge is live generalization. Continuous webcam windows contain idle poses, sign preparation, partial signs, and transitions, while the model learned from isolated, trimmed sign videos. The first continuous webcam trial therefore failed to predict `hat` and frequently reported insufficient hand-landmark coverage. A manual start/stop capture mode has now been implemented so that one captured webcam segment corresponds to one isolated sign. This mode needs controlled live testing and threshold calibration.

## 2. Problem and intended solution

Many Deaf and hard-of-hearing ISL users face communication barriers when interacting with people who do not understand sign language. The project investigates an affordable software approach that uses a normal webcam instead of gloves or specialised sensors.

The intended runtime flow is:

```text
Webcam video
    -> sign-sequence capture
    -> MediaPipe hand and upper-body landmarks
    -> normalization and temporal features
    -> compact Transformer
    -> confidence and unknown rejection
    -> recognized text
    -> optional speech
```

The current scope covers isolated signs. Continuous sentence recognition, automatic sign-boundary detection, and grammatical translation remain later work.

## 3. Dataset preparation

### INCLUDE-50 development benchmark

INCLUDE-50 provided the controlled first milestone:

| Split | Videos |
|---|---:|
| Train | 689 |
| Validation | 77 |
| Test | 192 |
| Total | 958 |
| Classes | 50 |

All 958 expected videos produced valid MediaPipe feature archives. The quality report found no missing or invalid manifest features. Twenty-one earlier pilot files were correctly identified as unexpected and excluded from model loading.

### Full INCLUDE benchmark

The official source contains 4,292 manifest entries across 263 classes. Eight unavailable videos all belonged to `Second (Number)`. The project documented this discrepancy and created a reproducible local benchmark that excludes this incomplete class:

| Split | Videos |
|---|---:|
| Train | 3,121 |
| Validation | 347 |
| Test | 816 |
| Total | 4,284 |
| Classes | 262 |

All 4,284 selected videos have valid landmark archives.

## 4. Landmark representation and quality controls

Each video is represented by a sequence of up to 64 timesteps. Every timestep contains 51 landmarks:

- 21 left-hand landmarks
- 21 right-hand landmarks
- 9 selected pose landmarks

The model input contains 255 values per timestep:

- 153 normalized coordinate values
- 51 confidence values
- 51 detected/not-detected indicators

The extraction pipeline provides:

- official-manifest-driven sample selection
- deterministic temporal sampling
- reusable MediaPipe timestamps
- split and class metadata
- resume support
- extraction and validation reports
- hand, pose, padding, and normalization coverage metrics

Normalization reduces variation caused by a person's position and scale in the frame. Padding masks prevent artificial padded timesteps from affecting Transformer attention.

## 5. Model experiments

The project compared three main model families using the same data splits and preprocessing.

### Support Vector Machine

The SVM summarizes every landmark sequence into a fixed feature vector. It is fast and establishes whether the landmarks contain class information, but it loses detailed movement order.

### Long Short-Term Memory network

The LSTM reads the sequence in order and retains temporal information through recurrent memory. It improved upon the SVM on the full benchmark, confirming that motion order matters.

### Compact Transformer

The Transformer uses self-attention to compare different moments in the sign directly. The compact configuration uses a 128-dimensional model, four attention heads, two encoder layers, a 256-dimensional feed-forward layer, and 0.3 dropout.

The larger Transformer performed worse than the compact model on validation data, showing that more parameters did not help this dataset.

## 6. Benchmark results

### INCLUDE-50 test results

| Model | Test accuracy | Test Macro F1 | Correct videos |
|---|---:|---:|---:|
| SVM | 67.19% | 63.94% | 129/192 |
| LSTM | 71.35% | 63.16% | 137/192 |
| Compact Transformer | **97.92%** | **97.83%** | **188/192** |

The Transformer validation Macro F1 remained between 97.60% and 100.00% across seeds 42, 7, and 21.

### Full INCLUDE test results

| Model | Validation Macro F1 | Test accuracy | Test Macro F1 | Correct videos |
|---|---:|---:|---:|---:|
| RBF SVM | 43.20% | 53.92% | 51.05% | 440/816 |
| LSTM | 62.48% | 64.83% | 64.01% | 529/816 |
| Compact Transformer | **91.65%** | **94.36%** | **94.47%** | **770/816** |

The compact Transformer improved full-INCLUDE test Macro F1 by 30.46 percentage points over the LSTM and 43.42 points over the SVM.

## 7. Why the compact Transformer performed well

The current evidence supports the following interpretation:

- Self-attention can compare start, middle, and end poses without compressing the complete movement into a single recurrent state.
- Two-hand coordination and relationships with upper-body pose remain available across the complete sequence.
- Landmark input removes most background, clothing, and texture variation before classification.
- Positional embeddings preserve movement order.
- Padding masks stop empty timesteps from influencing attention.
- The compact architecture provides enough capacity without the overfitting risk of the larger Transformer.

These explanations are consistent with the observed model comparison. Separate ablation experiments would be needed to measure each contribution independently.

## 8. Inference implementation completed

The following runtime components are implemented:

1. Checkpoint loader with architecture and metadata validation
2. In-memory landmark-sequence predictor
3. Unit tests for normalization, padding, top-k output, and malformed input
4. Single-video prediction script
5. Webcam landmark capture and visualisation
6. Hand-coverage and normalization-coverage checks
7. Probability and top-two margin rejection
8. Temporal smoothing and duplicate suppression for continuous-window experiments
9. JSONL logging for live trials and latency
10. Manual Space-key start/stop capture for isolated signs

The current automated test suite contains 70 passing tests.

## 9. Demonstrated results and current limitation

### Working result

The single-video inference script processed a known INCLUDE `hat` video through MediaPipe and the final Transformer. It reported:

- device: CUDA
- 64 of 64 real timesteps
- left-hand coverage: 100%
- right-hand coverage: 62.5%
- pose coverage: 100%
- top prediction: `hat`
- reported probability: 100%

### Failed continuous webcam trial

The first logged `hat` webcam session contained 363 prediction attempts:

- 252 rejected for insufficient hand coverage
- 100 rejected for low probability
- 11 still collecting
- zero `hat` top predictions
- zero emitted words
- average runtime rate of about 9.4 FPS

The failure showed a mismatch between isolated training clips and an uncontrolled rolling webcam window. A 64-frame live window at roughly 9.4 FPS spans almost seven seconds and may mix idle frames, preparation, the sign, a held pose, and return to rest.

### Implemented response

The webcam tool now supports isolated manual capture:

- Space starts a clean capture.
- The user performs one complete sign.
- Space stops capture and runs one prediction.
- Long captures are uniformly sampled across the complete motion.
- Short captures retain their real timesteps and use padding.
- The tool prints the top five predictions and the rejection reason.
- `R` discards a capture, `C` clears recognized text, and `Q` exits.

This solution matches the current isolated-sign model more closely. Controlled live trials remain pending.

## 10. Honest interpretation of progress

### Demonstrated

- Reproducible dataset and manifest handling
- Complete, validated landmark features for 4,284 videos
- Comparative SVM, LSTM, and Transformer benchmarks
- Strong locked test performance in the INCLUDE environment
- Working checkpoint loading and offline video prediction
- Working webcam landmark capture, logging, and manual sign segmentation

### Not yet demonstrated

- Reliable recognition for new users in uncontrolled webcam environments
- Calibrated unknown rejection thresholds
- Automatic start and end detection for signs
- Continuous sentence recognition
- FastAPI and frontend integration with the final model
- Text-to-speech in the completed application flow

The 94.47% Macro F1 result measures the official isolated-video benchmark. It must not be presented as 94.47% live-webcam accuracy.

## 11. Immediate next steps

1. Run controlled manual-capture trials for several signs and multiple repetitions.
2. Measure expected-label rank, probability, hand coverage, and capture duration.
3. Compare live captures with dataset examples to identify domain and signing differences.
4. Calibrate rejection thresholds using live-development samples, without reusing the frozen test split.
5. Measure MediaPipe and model latency separately on CUDA.
6. Implement automatic sign-boundary detection after manual capture becomes reliable.
7. Integrate the stable inference engine with FastAPI, the frontend, and optional speech output.
8. If landmarks remain the limiting factor, compare a compact pretrained CNN video baseline before considering a hybrid model.

## 12. Suggested presentation content and speaker notes

### Slide 1: Project progress

**On slide**

- Real-Time Indian Sign Language Communication Assistant
- Progress review
- Team and guide details from the existing project deck

**Speaker note:** The project focuses on isolated ISL recognition using a normal webcam, MediaPipe landmarks, and a trained temporal classifier.

### Slide 2: Problem and project scope

**On slide**

- Communication barrier between ISL users and non-signers
- Webcam-based recognition without wearable sensors
- Current scope: isolated signs converted to text
- Later scope: automatic segmentation, frontend, and speech

**Speaker note:** Clearly distinguish isolated recognition from continuous sign-language translation.

### Slide 3: End-to-end system

**On slide**

```text
Video capture -> MediaPipe -> normalized temporal features
-> compact Transformer -> rejection -> text
```

- 51 landmarks
- 255 values per timestep
- maximum 64 timesteps

**Speaker note:** MediaPipe converts images into compact hand and upper-body geometry. The Transformer classifies movement across time.

### Slide 4: Dataset and feature quality

**On slide**

- INCLUDE-50: 958 videos, 50 classes
- Full local INCLUDE: 4,284 videos, 262 classes
- 4,284 valid landmark archives
- Official train, validation, and test identities preserved

**Speaker note:** Explain the eight unavailable videos and the documented exclusion of the incomplete `Second (Number)` class.

### Slide 5: Model comparison

**On slide**

- SVM test Macro F1: 51.05%
- LSTM test Macro F1: 64.01%
- Compact Transformer test Macro F1: 94.47%
- Transformer test accuracy: 94.36%

**Speaker note:** The SVM loses detailed frame order, the LSTM keeps sequence memory, and Transformer attention compares distant frames directly.

### Slide 6: Why the Transformer won

**On slide**

- Global comparison across the sign sequence
- Direct modelling of two-hand coordination
- Normalized landmarks reduce background variation
- Compact model outperformed the larger configuration

**Speaker note:** Present these as evidence-based explanations, not as completed ablation results.

### Slide 7: Working inference result

**On slide**

- Known `hat` video classified correctly
- 64/64 timesteps
- 100% left-hand coverage
- 62.5% right-hand coverage
- `hat` ranked first with 100% reported probability

**Speaker note:** This proves that video decoding, MediaPipe extraction, checkpoint loading, feature normalization, and Transformer prediction work together.

### Slide 8: Webcam challenge and solution

**On slide**

- Continuous rolling window mixed idle, preparation, sign, and transition frames
- 252 of 363 attempts failed hand-coverage checks
- Zero `hat` top predictions in the first trial
- Manual Space-key start/stop capture now isolates one sign

**Speaker note:** The failed trial produced useful engineering evidence. The limitation comes from deployment distribution and sign segmentation, not merely the model checkpoint.

### Slide 9: Current status

**On slide**

**Complete:** data pipeline, feature validation, three-model benchmark, checkpoint inference, video prediction, webcam capture, logging, manual segmentation

**In progress:** controlled webcam trials, rejection calibration, latency evaluation

**Next:** automatic sign boundaries, API/frontend integration, text-to-speech

**Speaker note:** The system has moved from model research into deployment validation.

### Slide 10: Demonstration and next milestone

**On slide**

- Demo: known video prediction or manual webcam capture
- Benchmark: 94.47% test Macro F1 on 262 classes
- Next milestone: repeatable live recognition for unseen webcam users

**Speaker note:** If live conditions are unreliable during the presentation, use the known-video demonstration and show the manual webcam capture as current development work.

## 13. Presentation safety plan

For tomorrow's presentation:

1. Keep the known `hat` dataset video available as the reliable primary demo.
2. Keep the final Transformer checkpoint and CUDA environment ready.
3. Run one rehearsal using the exact presentation laptop and webcam.
4. Keep a screenshot or terminal capture of the successful known-video result.
5. Present manual webcam capture as a development prototype unless repeated tests confirm reliable output.
6. Do not retrain models or change thresholds immediately before the presentation.

## 14. Reproducible demonstration commands

### Reliable known-video demonstration

```powershell
uv run python scripts/predict_landmark_video.py `
  --video "E:\Hugging_Face\INCLUDE\raw\include\Clothes\37. Hat\MVI_3688.MOV" `
  --checkpoint "E:\Hugging_Face\INCLUDE\processed\mediapipe_v1\models\full_include\transformer_final_seed_42\selected_final.pt" `
  --device cuda
```

### Manual webcam capture trial

```powershell
uv run python scripts/run_webcam_inference.py `
  --checkpoint "E:\Hugging_Face\INCLUDE\processed\mediapipe_v1\models\full_include\transformer_final_seed_42\selected_final.pt" `
  --device cuda `
  --minimum-frames 16 `
  --expected-label "hat" `
  --trial-id "hat-manual-presentation" `
  --log "E:\Hugging_Face\INCLUDE\processed\mediapipe_v1\live_evaluation\hat-manual-presentation.jsonl"
```

In the webcam window, press Space to start, perform one sign, and press Space again to stop and classify.

## 15. Evidence sources inside the project

- `documents/stage 2/02-data-collection-report.md`
- `documents/stage 4/01-pilot-dataset-and-include50.md`
- `documents/stage 5/02-full-include-landmark-extraction.md`
- `documents/stage 6/01-include50-model-benchmark.md`
- `documents/stage 6/03-full-include-model-benchmark.md`
- `documents/stage 8/01-real-time-inference-implementation-order.md`
- `scripts/predict_landmark_video.py`
- `scripts/run_webcam_inference.py`
- `src/isl_assistant/inference/`
- `tests/`
