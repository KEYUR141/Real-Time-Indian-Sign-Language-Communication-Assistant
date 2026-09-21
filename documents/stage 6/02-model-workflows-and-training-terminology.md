# Model Workflows and Training Terminology

## 1. Purpose

This guide explains the three model families used by the Indian Sign Language
assistant and connects common machine-learning terminology to the project's
actual implementation. The benchmark compares:

1. a Support Vector Machine (SVM);
2. a Long Short-Term Memory network (LSTM);
3. a compact Transformer encoder.

All three models use the same manifest-selected MediaPipe landmark archives.
They differ in how they represent and learn from the movement sequence.

## 2. Shared Input Pipeline

Every source video follows this path:

```text
isolated-sign video
    -> MediaPipe hand and pose detection
    -> temporal sampling or padding to 64 frames
    -> coordinate normalization
    -> confidence and detection masks
    -> classifier
    -> one predicted sign class
```

Each frame contains 51 landmarks: 21 for each hand and 9 upper-body pose
landmarks. A temporal frame has 255 values:

- 153 normalized coordinates: 51 landmarks times x, y, and z;
- 51 confidence values;
- 51 detection indicators.

The temporal input shape for one video is therefore `64 x 255`. Padding masks
or true sequence lengths prevent artificial padded frames from being treated
as observed movement.

## 3. SVM Workflow

SVM means **Support Vector Machine**. It cannot directly consume a variable
movement sequence, so the 64 frames are summarized into one fixed
1,377-dimensional vector.

```text
64 x 255 temporal sequence
    -> position, movement, confidence, and detection statistics
    -> standardization
    -> RBF-kernel SVM
    -> class decision
```

The summary includes coordinate means, deviations, ranges, first and last
positions, velocity statistics, confidence statistics, and detection rates.
This is computationally inexpensive and provides an important classical
baseline. Its main limitation is that summarization loses much of the precise
frame order.

`C` controls the penalty for classification errors. A small `C` allows a
smoother boundary with more training errors; a large `C` tries harder to fit
the training samples and may overfit. The RBF kernel allows nonlinear class
boundaries.

SVM training is deterministic for the same data and library implementation;
the neural-network seed procedure is not required for this baseline.

## 4. LSTM Workflow

LSTM means **Long Short-Term Memory**. It is a recurrent neural network that
reads frames sequentially and updates an internal hidden state.

```text
frame 1 -> memory update
frame 2 -> memory update
...
final real frame -> sequence representation -> classifier
```

Input, forget, and output gates regulate what information is stored, removed,
and exposed. The true sequence length is supplied so padding does not update
the recurrent representation.

The LSTM preserves movement order, unlike the SVM summary. Its limitation is
that information from an early frame must travel through every later recurrent
step before it can influence the final representation.

## 5. Compact Transformer Workflow

The Transformer treats every real frame as a token. A learned linear layer
projects each 255-value frame to a 128-value embedding.

```text
64 x 255 frame features
    -> 64 x 128 frame embeddings
    -> add learned positions and a classification token
    -> two self-attention encoder layers
    -> classification-token representation
    -> 262 class scores
```

Positional embeddings preserve frame order. Self-attention allows any frame
to compare directly with any other frame, which is useful for start-to-end
movement, direction changes, two-hand coordination, and hand-to-body
relationships. The padding mask excludes artificial frames from attention.

The selected compact configuration uses:

| Setting | Value |
|---|---:|
| Model dimension | 128 |
| Attention heads | 4 |
| Encoder layers | 2 |
| Feed-forward dimension | 256 |
| Dropout | 0.3 |
| Learning rate | 0.001 |

A larger experiment used dimension 256, 8 heads, 4 layers, and feed-forward
dimension 1,024. It achieved 88.67% validation Macro F1, below the compact
model's three-seed mean of 90.83%. The larger model was therefore rejected.

## 6. Epochs, Batches, and Updates

An **epoch** is one complete pass through the training dataset. Full INCLUDE
contains 3,121 training videos. With a batch size of 32, one epoch contains
approximately 98 optimizer updates.

For every batch, the neural model:

1. produces class scores;
2. compares them with the correct labels using cross-entropy loss;
3. calculates gradients through backpropagation;
4. updates its parameters with AdamW.

More epochs provide more learning opportunities, but excessive training can
memorize the training data and reduce validation performance. `--epochs`
defines the maximum, not necessarily the selected epoch.

## 7. Validation, Early Stopping, and Patience

After every epoch, the model is evaluated on the validation split. Validation
Macro F1 selects the best checkpoint without consulting the test split.

**Patience** is the number of consecutive non-improving epochs allowed before
training stops. With patience 12, if no epoch beats the current best
validation Macro F1 for 12 epochs, training ends and restores the best state.

For example, the final compact Transformer reached its best validation result
at epoch 62. Later epochs did not replace that checkpoint. Final fitting then
used train plus validation data for exactly the already selected 62 epochs
before evaluating the test split once.

## 8. Random Seeds

A **seed** initializes pseudorandom operations such as:

- initial neural-network weights;
- training-batch shuffle order;
- dropout masks;
- some device-specific operations.

Seeds 7, 21, and 42 do not identify signs, classes, or videos. They create
different but reproducible training paths. Multiple seeds reveal whether an
architecture is stable or whether one result was unusually lucky.

Full-INCLUDE compact Transformer validation results were:

| Seed | Validation Macro F1 | Best epoch |
|---:|---:|---:|
| 7 | 92.11% | 53 |
| 21 | 89.54% | 32 |
| 42 primary final run | 91.65% | 62 |

The earlier seed-42 CPU run produced 90.84% at epoch 59. CPU and CUDA kernels
can follow slightly different numerical paths even with the same seed. The
three originally planned compact runs averaged approximately 90.83% with a
population standard deviation of approximately 1.05 percentage points.

Seed 42 remained the predefined primary run instead of selecting seed 7 merely
because it obtained the highest validation value.

## 9. Other Training Terms

| Term | Meaning in this project |
|---|---|
| Loss | Numerical training error minimized by the optimizer |
| Cross-entropy | Loss that rewards a high score for the correct sign class |
| Gradient | Direction and magnitude of a parameter adjustment |
| Backpropagation | Method used to calculate gradients through the network |
| AdamW | Optimizer used for LSTM and Transformer parameters |
| Learning rate | Size of each optimizer step |
| Dropout | Randomly disables activations during training to reduce overfitting |
| Hidden/model size | Width of the learned internal representation |
| Attention head | One learned view of relationships between frames |
| Feed-forward layer | Per-token nonlinear transformation after attention |
| Class weighting | Increases the influence of underrepresented training classes |
| Gradient clipping | Limits extreme recurrent or Transformer gradient updates |
| Checkpoint | Saved parameters and metadata from a particular trained state |
| Overfitting | Improving on training data while generalization becomes worse |

## 10. Metrics and Dataset Splits

The training split updates model parameters. The validation split selects
configuration and stopping epoch. The test split estimates final performance
only after all choices are locked.

Accuracy is the proportion of correct videos. Macro F1 calculates F1 for every
class and gives each class equal importance. Macro F1 is the primary selection
metric because a model should not hide weak classes behind stronger or more
frequent classes.

Sixteen of the 262 retained classes have no validation example, which means
their validation behavior cannot be measured directly. All 262 classes are
represented in the test split.

## 11. What a `.pt` File Means

`.pt` is a conventional filename extension for a file serialized by PyTorch.
It is not a different neural architecture and does not mean plain text.

This project's validation `.pt` files contain:

- architecture and model dimensions;
- number of output classes;
- trained `state_dict` parameter tensors;
- train-derived normalization mean and standard deviation;
- label-ID-to-name mapping;
- Transformer configuration where applicable.

`selected_final.pt` contains the locked model refitted on train plus validation
data. Loading it requires reconstructing the recorded architecture, loading the
`state_dict`, applying the stored normalization statistics, and using the
stored label mapping to translate output IDs into sign names.

PyTorch checkpoint files should be loaded only from trusted sources because
general serialized model files can be unsafe when obtained from an unknown
party. The checkpoint should be versioned together with its preprocessing
schema and benchmark report.

## 12. Locked Full-INCLUDE Transformer Result

The selected compact Transformer was evaluated once on the complete official
local test split:

| Metric | Result |
|---|---:|
| Classes | 262 |
| Train samples | 3,121 |
| Validation samples | 347 |
| Test samples | 816 |
| Validation accuracy | 92.80% |
| Validation Macro F1 | 91.65% |
| Selected epoch | 62 |
| Test accuracy | 94.36% |
| Test Macro F1 | 94.47% |
| Correct test predictions | 770/816 |

The Transformer result is frozen. Future decisions must not repeatedly use its
test errors for model selection.

