# Real-Time Indian Sign Language Communication Assistant

Team preparation notes for the final presentation and viva. Commands are in
[cmd.txt](cmd.txt). The same study notes are available in [presentation.txt](presentation.txt).

## 1. A short opening

Our project recognizes one isolated Indian Sign Language sign at a time from
an uploaded video or a manually captured webcam sequence. Google MediaPipe
extracts hand and upper-body landmarks. We process those landmarks and use a
trained classifier to predict a sign label. We compared SVM, LSTM, and a
Compact Transformer; GRU was an additional temporal comparison. The Compact
Transformer performed best on our recorded INCLUDE benchmark.

This is a fixed-vocabulary isolated-sign recognition prototype. The displayed
word is a class label, not a complete grammatical translation of a conversation.

## 2. Five-minute presentation and team order

| Time | Speaker | What to cover |
|---|---|---|
| 0:00-0:45 | Keyur Sherke | Problem, objective, and scope |
| 0:45-1:45 | Divya Lonare | Dataset, MediaPipe, preprocessing, system flow |
| 1:45-3:00 | Lucky Dhepe | Models, why compare them, metrics, result table |
| 3:00-4:30 | Alkesh Soni | Frontend/API demonstration and observed limitations |
| 4:30-5:00 | Keyur Sherke | Conclusion and future scope |

This is a suggested speaking order, not a claim about individual development
contributions. Every member should understand the full flow and result caveat.

## 3. Important full forms

| Term | Full form or meaning in this project |
|---|---|
| ISL | Indian Sign Language |
| ML | Machine Learning |
| DL | Deep Learning |
| SVM | Support Vector Machine |
| RBF | Radial Basis Function, the SVM kernel used here |
| RNN | Recurrent Neural Network, the family containing GRU and LSTM |
| GRU | Gated Recurrent Unit |
| LSTM | Long Short-Term Memory |
| Transformer | An architecture name, not an acronym; we use a compact encoder |
| CNN | Convolutional Neural Network; no separate RGB CNN classifier was benchmarked here |
| MLP | Multilayer Perceptron; a general term for feed-forward neural layers |
| CPU / GPU | Central Processing Unit / Graphics Processing Unit |
| CUDA | Compute Unified Device Architecture; NVIDIA's GPU computing platform |
| API | Application Programming Interface |
| HTTP | Hypertext Transfer Protocol |
| JSON | JavaScript Object Notation; API responses and reports |
| FPS | Frames Per Second |
| TP / FP / FN / TN | True Positive / False Positive / False Negative / True Negative |
| F1 | Name of the harmonic-mean metric; it is not a model or an acronym |
| .npz | NumPy archive holding landmark arrays and metadata |
| .pt | Conventional PyTorch checkpoint extension; stores weights and supporting data |

MediaPipe is Google's perception toolkit. FastAPI and React are framework names.
PyTorch is our neural-model framework; scikit-learn supplies the SVM pipeline.

## 4. Dataset and the actual benchmark scope

INCLUDE is the video dataset obtained from AI4Bharat's published resources.
Our manifests list 4,292 videos across 263 official classes. The local benchmark
excludes eight unavailable videos from the `Second (Number)` class, leaving
4,284 usable videos across 262 classes. Say "the locally available INCLUDE
benchmark" when describing the larger experiment.

| Dataset | Train | Validation | Test | Total | Classes |
|---|---:|---:|---:|---:|---:|
| INCLUDE-50 | 689 | 77 | 192 | 958 | 50 |
| Local full INCLUDE | 3,121 | 347 | 816 | 4,284 | 262 |

INCLUDE-50 was the pilot benchmark before scaling to the larger class set.
Kaggle and ISLVT downloads were explored during collection, but they were not
included in these reported training/test results. Do not add their sample
counts to the INCLUDE benchmark or describe them as training data for it.

## 5. The system flow in simple language

```text
OFFLINE TRAINING
INCLUDE videos + official train/validation/test manifests
  -> Google MediaPipe detects hands and upper-body points
  -> Sample up to 64 frames and pad shorter sequences
  -> Center coordinates on shoulders and scale by shoulder width
  -> Save .npz landmarks, masks, confidence, labels, splits and timestamps
  -> Validate files and audit manifest/class coverage
  -> Train SVM / GRU / LSTM / Compact Transformer
  -> Select using validation Macro F1
  -> Refit the chosen configuration on train + validation
  -> Evaluate on held-out test data and save model/report artifacts

APPLICATION: UPLOADED VIDEO
React upload -> FastAPI -> backend MediaPipe extraction
  -> matching preprocessing -> loaded Transformer -> probabilities
  -> acceptance checks -> sign label or unknown -> React result

APPLICATION: BROWSER WEBCAM
React camera -> browser MediaPipe + visible connected points
  -> capture one complete gesture -> send landmark sequence to FastAPI
  -> sampling + normalization -> loaded Transformer
  -> quality/confidence checks -> sign label or unknown -> React result
```

Browser live points show landmark detection. The Python model makes the sign
classification. The webcam branch sends points directly instead of requiring
OpenCV to decode a browser recording. Desktop webcam capture is also available.

## 6. What the model receives

Each sampled frame has 51 landmarks: 21 left-hand points, 21 right-hand points,
and nine selected upper-body pose points. Face landmarks are not included in
this trained feature schema.

Each temporal frame contains 255 values: 153 coordinates (51 x 3), 51 confidence
values, and 51 detection indicators. Neural input is up to 64 x 255 values per
video. Sixty-four is a sequence length, not the number of classes or camera FPS.

Shoulder-based normalization reduces position/scale differences; it does not
guarantee invariance to new users, camera angles, signing styles, or occlusion.
Short clips are padded, with true lengths and attention masks protecting the
models from treating padding as real movement. Statistical standardization is
fit on training data during selection, then on train + validation for final refit.

## 7. How the models work

### SVM: Support Vector Machine

The SVM summarizes each gesture into 1,377 features: coordinate means, variation,
minimum/maximum, starting/ending positions, movement differences, confidence,
and detection rates. A StandardScaler and RBF SVM learn boundaries between
sign classes. Validation selected C = 10 from 0.1, 1, and 10.

Flow: sequence -> summary vector -> scale -> RBF decision scores -> class.
It is a useful classical baseline, but summarization loses detailed movement
order. This scikit-learn trainer runs on the CPU, not through PyTorch CUDA.

### GRU: Gated Recurrent Unit

The GRU reads frames in order and updates a hidden memory using reset and update
gates. Our implementation classifies the final hidden state, with hidden size
128 and dropout before the output layer. It was an additional INCLUDE-50
validation comparison, not a fourth locked full-INCLUDE test result.

Flow: ordered frames -> gated hidden-state updates -> final state -> class.

### LSTM: Long Short-Term Memory

The LSTM also reads the sequence in order. Input, forget, and output gates
control a cell state and hidden state, helping retain useful earlier information.
Our single-layer LSTM has hidden size 128; classification uses its final hidden
state with dropout. The full benchmark selected epoch 80 with a 140-epoch cap
and patience 20, then refitted for the selected epoch count.

Flow: ordered frames -> cell/hidden memory and gates -> final hidden state -> class.

### Compact Transformer

The model projects each 255-value frame to 128 dimensions, adds learned frame
positions and a classification token, and passes the sequence through two
encoder layers. Four attention heads compare relationships across frames.
The feed-forward dimension is 256 and dropout is 0.3. The output layer maps
the classification token to sign scores; softmax produces ranked probabilities.

Flow: frames -> 128-dimensional embeddings + positions + class token
-> masked self-attention encoders -> class-token representation -> sign scores.

It is an encoder classifier trained on landmarks, not a language model that
generates the next word. It does not download knowledge about sign meanings
from ChatGPT. It learns associations between labeled motion patterns and classes.

Attention can relate the start, middle, and end of a sign and both hands' motion.
Structured landmarks reduce the need to learn visual appearance from pixels.
These are plausible explanations for the observed performance, not proven
causes without ablation experiments. A larger tested Transformer achieved
88.67% validation Macro F1, below the selected compact configuration.

## 8. Training terms you should be able to explain

- Epoch: one pass through the training samples. It is not one frame or one class.
- Batch: a group processed before one optimizer update; default batch size is 32.
- Seed: controls random initialization, shuffling and dropout. Seeds 7, 21 and 42
  are experiment settings, not class IDs. CUDA runs can still vary.
- Learning rate: update step size; our compact default is 0.001 with AdamW.
- Loss: training error objective; neural trainers use class-weighted cross-entropy.
- Dropout: randomly disables activations during training to reduce overfitting;
  it is disabled during inference.
- Early stopping/patience: stop after enough epochs without improved validation
  Macro F1. Restore the best validation checkpoint, not simply the last epoch.
- Overfitting: learning training-specific patterns that fail to generalize.
- Checkpoint: saved parameters and architecture/preprocessing/label metadata.
- Training versus inference: training adjusts parameters; inference loads the
  saved model and predicts without training it again.
- Validation versus test: validation selects configurations; test measures the
  locked configuration. Our scripts refit on train + validation before final test.

## 9. Metrics with one worked example

For one class, treat `hat` as positive and every other sign as negative. Imagine
20 attempts, with this confusion matrix:

| Actual / predicted | Hat | Another sign |
|---|---:|---:|
| Hat | 8 (TP) | 4 (FN) |
| Another sign | 2 (FP) | 6 (TN) |

- Accuracy = correct / all = (8 + 6) / 20 = 70%. How often was any label correct?
- Precision for hat = TP / (TP + FP) = 8 / 10 = 80%. When we said hat, how often
  were we right?
- Recall for hat = TP / (TP + FN) = 8 / 12 = 66.67%. How many real hat signs
  did we find?
- F1 for hat = 2PR / (P + R) = 72.73%. It balances precision and recall.
- Support = actual examples of a class; hat has support 12 in this example.

Macro F1 computes F1 for each class and gives every class equal weight. For
example, class F1 values 0.90, 0.60, and 0.30 average to Macro F1 = 0.60.
Weighted F1 weights each class by support; frequent classes influence it more.
Multiclass confusion matrices show actual labels in rows and predicted labels
in columns. Diagonal entries are correct; other cells show confused signs.

The definitions are identical for SVM, GRU, LSTM and Transformer. They are not
different kinds of accuracy for different architectures. Accuracy and Macro F1
can disagree because class counts and error distributions differ.

Confidence is a probability for one prediction, not measured dataset accuracy.
A displayed 100% probability does not prove a prediction is correct, calibrated,
or evidence of universal 100% recognition. Missing-hand coverage is a tracking
quality measure, not a classification metric. Latency is processing time, not F1;
measure it on the demo machine, including warm-up and the chosen input branch.

## 10. Results to present

### Locked full local INCLUDE results

| Model | Validation accuracy | Validation Macro F1 | Test accuracy | Test Macro F1 | Correct test videos |
|---|---:|---:|---:|---:|---:|
| RBF SVM | 50.43% | 43.20% | 53.92% | 51.05% | 440/816 |
| LSTM | 68.01% | 62.48% | 64.83% | 64.01% | 529/816 |
| Compact Transformer | 92.80% | 91.65% | 94.36% | 94.47% | 770/816 |

The selected full-INCLUDE Transformer used seed 42 and selected epoch 62.
It made 46 errors out of 816 test clips. Its Macro F1 improvement over LSTM
was 30.46 percentage points. These are recorded experiment results, not values
recomputed by the frontend's benchmark panel.

### INCLUDE-50 pilot results

| Model | Validation Macro F1 | Test accuracy | Test Macro F1 |
|---|---:|---:|---:|
| SVM | 70.28% | 67.19% | 63.94% |
| GRU | 75.87% | Not independently tested | Not independently tested |
| LSTM | 76.74% | 71.35% | 63.16% |
| Compact Transformer | 97.60% | 97.92% | 97.83% |

Do not mix INCLUDE-50 validation results with the 262-class test results.
The pilot Transformer classified 188 of 192 test videos correctly. Additional
pilot seeds checked initialization sensitivity; they were not tests of classes
called happy or hot.

## 11. What the live demo proves and what remains limited

The React interface, FastAPI connection, landmark display, clip prediction, and
manual webcam capture are implemented. Inference loads the checkpoint once
and reuses it; it does not run dozens of training epochs for a prediction.
The first request includes loading overhead, so rehearse and warm up first.

The benchmark demonstrates performance on the selected official dataset splits.
It does not establish signer-independent accuracy or reliable continuous
conversation. New webcam users have produced unknown or wrong predictions.
Possible causes include unseen signing style, camera/view changes, incomplete
gesture boundaries, missing landmarks, and differences between training and
live detection. These need controlled experiments, not a claim that one cause
has already been proved.

Browser/desktop isolated capture checks sequence length, hand visibility,
normalization coverage, top probability, and the gap between the top two classes.
Uploaded-video acceptance checks are implemented separately with different
coverage criteria. Rejected input is shown as unknown; it is a decision policy,
not a trained universal unknown-sign class. Accepted predictions can still be wrong.

Temporal smoothing and duplicate-suppression helpers exist, but the current
desktop webcam demonstration classifies manually bounded isolated captures;
do not describe it as automatic continuous sign segmentation.

## 12. Suggested demonstration and closing wording

1. Start API and frontend using cmd.txt; check the checkpoint and CUDA.
2. Explain the dataset result panel and identify it as stored benchmark results.
3. Upload an identified INCLUDE test clip and show label, probabilities, coverage,
   and reported request latency. Say it is a dataset clip.
4. Show live points, capture one rehearsed isolated gesture, and discuss the
   actual result. If unknown appears, explain the rejection reason honestly.
5. Finish with the result and next scope, rather than retraining on stage.

Conclusion to speak: "We built an end-to-end prototype for isolated ISL sign
recognition. On our 262-class local INCLUDE benchmark, the Compact Transformer
achieved 94.36% test accuracy and 94.47% Macro F1, outperforming the SVM and
LSTM baselines. Our next priority is reliable recognition for new webcam users
through diverse recordings, consistent capture, and independent evaluation."

Future scope: collect consented multi-user webcam clips, align preprocessing,
evaluate by signer and environment, calibrate rejection on development data,
measure end-to-end latency, and study continuous segmentation. RGB/CNN or
hybrid features could be evaluated if landmark-only inputs miss necessary cues.

## 13. Short viva answers

- Why not only a single image? Movement and order distinguish many signs.
- Why MediaPipe? It provides compact hand/pose representations and visible tracking.
- Why not a huge Transformer? The compact one performed better in our experiments;
  larger capacity does not guarantee better generalization with limited samples.
- Do all models vote together in the application? No. They were compared offline;
  the deployed demo uses the selected Compact Transformer checkpoint.
- Do we use a decoder? No. This implementation is a Transformer encoder classifier.
- Why CUDA? It lets PyTorch execute neural computations on the NVIDIA GPU;
  CPU-only PyTorch cannot do that. CUDA is not required by the SVM trainer.
- Is it ready to replace an interpreter? No. Scope is isolated vocabulary and
  new-user reliability remains incomplete.
- Why is a test video prediction strong but my webcam sign wrong? The benchmark
  and new-user environments differ; both complete capture and generalization matter.
- Is the webcam result hidden in the benchmark accuracy? No. These evaluations
  are separate; no numerical live-user accuracy is established here.
- Does training all neural models together test all of them? No. The current
  combined trainer tests only the validation winner with --evaluate-test.
  Independent model runs are provided in cmd.txt.

## 14. Evidence in the repository

- [Full INCLUDE benchmark](documents/stage%206/03-full-include-model-benchmark.md)
- [INCLUDE-50 benchmark](documents/stage%206/01-include50-model-benchmark.md)
- [Training terminology](documents/stage%206/02-model-workflows-and-training-terminology.md)
- Model code: src/isl_assistant/models/temporal.py
- Training: scripts/train_svm_baseline.py and scripts/train_temporal_baselines.py
- Inference: src/isl_assistant/inference/temporal_predictor.py
- API routes and actual request policy: src/isl_assistant/api/
- Frontend wiring: frontend/README.md and frontend/vite.config.js

The saved JSON experiment reports and confusion matrices under the configured
external feature root are the experiment artifacts. Do not state a current
test-suite count or a universal latency unless it has been measured for that run.
