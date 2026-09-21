# INCLUDE-50 Landmark Model Benchmark

## 1. Purpose

This report records the first reproducible model benchmark for the Real-Time
Indian Sign Language Communication Assistant. It compares a classical SVM
baseline with GRU, LSTM, and compact Transformer sequence models on the
official INCLUDE-50 split.

The benchmark answers three questions:

1. Do the extracted MediaPipe landmarks contain enough information to classify
   the 50 selected signs?
2. Does preserving frame order improve performance over summarized features?
3. Which architecture should be carried forward to the full INCLUDE benchmark?

The definitions of accuracy, precision, recall, F1, Macro F1, and confusion
matrices are documented in
[ML Evaluation Metrics](../stage%205/01-ml-evaluation-metrics.md).

## 2. Dataset and Feature Integrity

INCLUDE-50 contains 958 official manifest samples across 50 classes.

| Split | Samples |
|---|---:|
| Train | 689 |
| Validation | 77 |
| Test | 192 |
| Total | 958 |

The manifest-aware audit found:

- 958 valid manifest feature archives;
- 0 missing manifest features;
- 0 invalid manifest features;
- the exact expected train, validation, and test counts;
- 21 earlier pilot files, correctly reported as unexpected and excluded from
  every model loader.

Each video was represented as a maximum 64-frame landmark sequence containing
51 landmarks:

- 21 left-hand landmarks;
- 21 right-hand landmarks;
- 9 selected pose landmarks.

The temporal input at each frame contained 255 values:

- 153 normalized coordinates: 51 landmarks times 3 coordinates;
- 51 MediaPipe confidence values;
- 51 landmark-detection indicators.

Short videos were right-padded. GRU and LSTM used true sequence lengths, while
the Transformer used an attention padding mask. Automated tests verified that
changing values in padded Transformer frames does not change its prediction.

## 3. Evaluation Protocol

Macro F1 was chosen as the primary selection metric because every sign class
must matter equally, including classes with fewer evaluation examples.

The experimental protocol was:

1. Fit model parameters using the training split.
2. Select hyperparameters, architecture, and stopping epoch using validation
   Macro F1.
3. Refit the selected configuration using train plus validation samples.
4. Evaluate the locked model on the test split.

The SVM is deterministic for the same data and configuration. Neural models
contain randomness in initialization, dropout, and batch order. The Transformer
was therefore checked using seeds 42, 7, and 21 before final test evaluation.

## 4. Model Configurations

### 4.1 SVM

The Support Vector Machine used an RBF kernel with balanced class weights.
Each sequence was converted into a 1,377-dimensional vector containing:

- coordinate mean, standard deviation, minimum, and maximum;
- first-frame and last-frame coordinates;
- velocity mean and standard deviation;
- confidence statistics;
- landmark detection rates.

Validation selected C = 10 from candidates 0.1, 1, and 10.

### 4.2 GRU

The Gated Recurrent Unit processed the full ordered sequence using:

- hidden dimension 128;
- dropout 0.3 before classification;
- class-weighted cross-entropy;
- AdamW optimization;
- gradient clipping;
- validation-Macro-F1 early stopping.

Its best validation result occurred at epoch 50.

### 4.3 LSTM

The Long Short-Term Memory model used the same input, hidden dimension,
optimization, class weighting, and stopping protocol as the GRU.

Its best validation result occurred at epoch 63.

### 4.4 Compact Transformer

The Transformer used:

- input projection from 255 to model dimension 128;
- a learned classification token;
- learned positional embeddings;
- 2 Transformer encoder layers;
- 4 attention heads;
- feed-forward dimension 256;
- dropout 0.3;
- GELU activation;
- attention masks for padded frames;
- class-weighted cross-entropy;
- AdamW optimization and gradient clipping.

The seed-42 model selected epoch 47.

## 5. Validation Results

| Model | Validation accuracy | Validation Macro F1 | Best epoch |
|---|---:|---:|---:|
| SVM | 75.32% | 70.28% | Not applicable |
| GRU | 75.32% | 75.87% | 50 |
| LSTM | 77.92% | 76.74% | 63 |
| Compact Transformer, seed 42 | 98.70% | 97.60% | 47 |

The Transformer was repeated to determine whether seed 42 was unusually lucky.

| Transformer seed | Validation accuracy | Validation Macro F1 | Best epoch |
|---:|---:|---:|---:|
| 7 | 100.00% | 100.00% | 41 |
| 21 | 97.40% | 97.60% | 43 |
| 42 | 98.70% | 97.60% | 47 |

Across the three seeds:

- mean validation Macro F1: 98.40%;
- population standard deviation: 1.13 percentage points;
- minimum: 97.60%;
- maximum: 100.00%.

This shows that the strong validation result was stable across the tested
random initializations.

## 6. Test Results

| Model | Test accuracy | Test Macro F1 | Correct predictions |
|---|---:|---:|---:|
| SVM | 67.19% | 63.94% | 129/192 |
| LSTM | 71.35% | 63.16% | 137/192 |
| Compact Transformer | **97.92%** | **97.83%** | **188/192** |

GRU was retained as a validation comparison and was not independently evaluated
on the test split. This avoided unnecessary repeated use of the test data.

Although LSTM increased raw accuracy over SVM, it slightly reduced Macro F1.
It improved some signs while becoming substantially worse on others. Therefore,
it did not replace SVM under the predefined primary metric.

The Transformer improved both accuracy and Macro F1 by a large margin and was
the clear INCLUDE-50 winner.

## 7. Transformer Test Errors

Only four Transformer test videos were misclassified:

| Actual sign | Predicted sign | Count |
|---|---|---:|
| hello | girl | 2 |
| year | paint | 1 |
| boy | father | 1 |

The lowest non-perfect class F1 scores were:

| Class | Precision | Recall | F1 | Test support |
|---|---:|---:|---:|---:|
| girl | 0.50 | 1.00 | 0.67 | 2 |
| father | 0.67 | 1.00 | 0.80 | 2 |
| hello | 1.00 | 0.71 | 0.83 | 7 |
| boy | 1.00 | 0.75 | 0.86 | 4 |
| paint | 0.75 | 1.00 | 0.86 | 3 |
| year | 1.00 | 0.86 | 0.92 | 7 |

All other represented test classes achieved F1 = 1.00.

## 8. Why the Transformer Performed Well

The following explanations are supported by the architecture and observed
results, although ablation studies are required to measure each contribution.

### 8.1 Direct access to the complete movement

GRU and LSTM process frames sequentially and compress earlier information into
a recurrent hidden state. Self-attention allows the Transformer to compare any
frame directly with any other frame. This is useful for signs distinguished by:

- start and end hand position;
- movement direction;
- changes of direction;
- coordination between both hands;
- relationships between hands, face, shoulders, and torso.

### 8.2 Structured landmarks instead of raw pixels

The Transformer received compact MediaPipe features rather than images. It did
not need to learn to ignore backgrounds, clothing, lighting, or image texture.
Most of its capacity could be used to model body configuration and movement.

### 8.3 Position and padding were represented correctly

Learned positional embeddings preserved frame order. The attention mask
prevented padded frames from participating in attention. This gave the model
the complete ordered movement without confusing artificial padding with real
motion.

### 8.4 Preprocessing reduced nuisance variation

Landmark normalization reduced differences in position and scale. Train-derived
standardization placed coordinate, confidence, and detection features on
comparable numerical scales.

### 8.5 Appropriate model size

The model was deliberately compact. Two encoder layers and four heads supplied
enough capacity for the 64-frame sequences without using a large
language-model-style Transformer.

### 8.6 Controlled isolated-sign task

Each video contains one isolated sign and one label. This is easier than
continuous sign-language recognition, which must also locate sign boundaries
and model grammatical context.

### 8.7 Stable result across random seeds

Validation Macro F1 remained between 97.60% and 100.00% across three seeds.
The final test Macro F1 of 97.83% closely matched validation performance,
providing stronger evidence that the result was not a lucky initialization.

## 9. Limitations and Interpretation

The 97.83% result is valid for the official INCLUDE-50 test split. It must not
be interpreted as guaranteed accuracy for unseen webcam users.

Important limitations are:

- only 77 validation videos, often one or two per class;
- possible signer or recording-condition overlap across official splits;
- isolated signs rather than continuous signing;
- controlled dataset recordings rather than live webcam input;
- confidence and detection patterns may encode dataset-specific information;
- no signer-independent benchmark has yet been confirmed.

The Transformer test result is now frozen. Future changes must not be selected
using these four test errors.

Recommended later ablations include:

- coordinates only versus coordinates plus confidence and detection;
- hands only versus hands plus pose;
- original order versus shuffled frame order;
- signer-independent evaluation;
- live webcam and out-of-distribution testing.

## 10. Reproducibility and Automated Tests

The implementation includes tests for:

- manifest parsing and nested Extra directories;
- manifest-safe feature paths and metadata;
- temporal sampling and padding;
- landmark normalization;
- data-quality reports;
- SVM summary feature construction;
- temporal dataset loading;
- GRU, LSTM, and Transformer output shapes;
- Transformer padded-frame invariance.

At the completion of the compact Transformer implementation, all 53 automated
tests passed and the selected source files passed lint and compilation checks.

Model artifacts and reports are stored outside the repository under the
configured processed-data root. They include validation checkpoints, final
model checkpoints, JSON metric reports, and CSV confusion matrices.

## 11. Conclusion

The complete landmark-to-classification pipeline is validated on INCLUDE-50.
SVM established a meaningful classical baseline. GRU and LSTM showed that
temporal information was useful, but neither clearly improved the primary test
metric. The compact Transformer produced the strongest and most stable result:

- 98.40% mean validation Macro F1 across three seeds;
- 97.92% final test accuracy;
- 97.83% final test Macro F1;
- 188 of 192 test videos correctly classified.

The compact Transformer is the selected architecture for expansion to full
INCLUDE.

## 12. Transition to the Full INCLUDE Benchmark

The full official INCLUDE manifests contain:

- 4,292 manifest entries;
- 263 official classes.

The local collection currently contains:

- 4,284 available videos;
- 262 populated classes;
- 8 missing videos, all belonging to Second (Number).

Before full-INCLUDE landmark extraction, the project must make and document one
of two choices:

1. recover all eight missing Second (Number) videos; or
2. explicitly exclude that class and its eight manifest entries, creating a
   reproducible 262-class local benchmark.

After that decision, the full benchmark should:

1. generate a versioned full-INCLUDE manifest;
2. extract landmarks with the unchanged MediaPipe schema;
3. audit exact manifest coverage and feature quality;
4. train SVM, LSTM, and compact Transformer configurations;
5. select models using validation Macro F1;
6. evaluate each locked benchmark configuration on test;
7. compare scaling, per-class behavior, training cost, and inference latency
   against the INCLUDE-50 results recorded here.
