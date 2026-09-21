# Full INCLUDE Model Benchmark

## 1. Benchmark Scope

This report records the locked SVM, LSTM, and compact Transformer results on
the locally complete portion of the full INCLUDE benchmark.

The unavailable `Second (Number)` class was excluded before label IDs were
reindexed. The resulting benchmark contains:

| Split | Samples |
|---|---:|
| Train | 3,121 |
| Validation | 347 |
| Test | 816 |
| Total | 4,284 |
| Classes | 262 |

Every selected video has a valid 64-frame MediaPipe landmark archive. All
models used the same official split identities and active full-INCLUDE label
map. Test evaluation occurred only after validation-based configuration and
epoch selection.

## 2. Locked Results

| Model | Validation accuracy | Validation Macro F1 | Test accuracy | Test Macro F1 | Correct test videos |
|---|---:|---:|---:|---:|---:|
| RBF SVM | 50.43% | 43.20% | 53.92% | 51.05% | 440/816 |
| LSTM | 68.01% | 62.48% | 64.83% | 64.01% | 529/816 |
| Compact Transformer | **92.80%** | **91.65%** | **94.36%** | **94.47%** | **770/816** |

The compact Transformer is the clear winner, improving test Macro F1 by 30.46
percentage points over LSTM and 43.42 points over SVM.

## 3. SVM Result

The SVM used 1,377 summary features per video, an RBF kernel, balanced class
weights, and validation-selected `C = 10`.

Its 51.05% test Macro F1 shows that the landmarks contain useful
class-discriminating information even after temporal summarization. The gap to
the sequence models also shows that precise movement order is important for
the 262-class problem.

The locked artifact is `svm_baseline.joblib`, accompanied by the JSON report
and test confusion matrix.

## 4. LSTM Result

The LSTM used hidden size 128, dropout 0.3, class-weighted cross-entropy,
AdamW, and seed 42 on CUDA. Training allowed 140 epochs with patience 20.

The best validation checkpoint occurred at epoch 80. Training ended at epoch
100 after 20 further epochs failed to improve Macro F1. The final model was
then refitted on train plus validation for the selected 80 epochs.

LSTM improved test Macro F1 from the SVM's 51.05% to 64.01%, demonstrating the
value of preserving frame order. Its recurrent compression was nevertheless
far less effective than global self-attention on this benchmark.

The locked artifact is `selected_final.pt`, accompanied by the validation
checkpoint, JSON report, and test confusion matrix.

## 5. Transformer Selection

The compact Transformer used model dimension 128, four attention heads, two
encoder layers, feed-forward dimension 256, dropout 0.3, and seed 42 for the
primary final result.

Additional compact seeds produced validation Macro F1 values of 92.11% for
seed 7 and 89.54% for seed 21. The originally planned three compact runs were
stable around a mean of 90.83% with approximately 1.05 percentage points of
population standard deviation.

A larger Transformer used model dimension 256, eight heads, four layers,
feed-forward dimension 1,024, and dropout 0.4. It achieved only 88.67%
validation Macro F1 and was rejected. Increasing capacity did not compensate
for the small number of training examples per class.

The final compact Transformer selected epoch 62, refitted on train plus
validation, and reached 94.47% test Macro F1. It correctly classified 770 of
816 videos. All 262 classes were represented in the test split; only
`afternoon`, with one test example, received zero recall.

The locked artifact is `selected_final.pt`, accompanied by the validation
checkpoint, JSON report, and test confusion matrix.

## 6. Interpretation

The results establish a consistent progression:

```text
summary statistics (SVM)
    < sequential memory (LSTM)
    < global temporal attention (Transformer)
```

The Transformer can directly compare distant frames and coordinate both hands
with upper-body pose. MediaPipe normalization removes much visual nuisance
variation, allowing the compact model to devote capacity to movement rather
than backgrounds and appearance.

The 94.47% result applies to the official INCLUDE environment. It does not by
itself establish signer-independent or live-webcam performance. Cross-dataset
and real-time evaluation remain necessary.

## 7. Frozen-Test Decision

All three test results are now frozen. Future architecture, preprocessing, or
hyperparameter changes must be selected using training and validation data,
not by repeatedly choosing the best score on these 816 test videos.

The compact Transformer is selected for deployment work. SVM and LSTM remain
documented reference baselines.

