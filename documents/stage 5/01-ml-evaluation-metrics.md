# ML Evaluation Metrics for the ISL Classifier

This guide explains the scores used to compare the SVM baseline and later
sequence models such as GRU, LSTM, and a Transformer. The metrics are the same
for every model; they measure predictions, not the model architecture.

## Example

Assume there are 20 real videos of the sign loud.

- The model predicts loud correctly for 16 videos.
- It predicts another sign for 4 real loud videos.
- It incorrectly calls 2 videos of other signs loud.

| Term | Meaning | Example |
|---|---|---:|
| True Positive (TP) | Correctly predicted loud | 16 |
| False Negative (FN) | Real loud predicted as another sign | 4 |
| False Positive (FP) | Another sign predicted as loud | 2 |

## Accuracy

Accuracy is the fraction of all predictions that are correct.

    accuracy = correct predictions / all predictions

If 85 of 100 test videos are correctly classified, accuracy is 85%. Accuracy is
easy to read, but it can hide weak performance on individual signs.

## Precision

Precision asks: when the model says a video is loud, how often is it right?

    precision = TP / (TP + FP)

In the example, precision is 16 / (16 + 2) = 88.9%. High precision means few
false alarms for that sign.

## Recall

Recall asks: of all real loud videos, how many did the model identify?

    recall = TP / (TP + FN)

In the example, recall is 16 / (16 + 4) = 80%. High recall means the model
does not miss many real examples.

## F1 Score

F1 balances precision and recall.

    F1 = 2 * (precision * recall) / (precision + recall)

The example gives an F1 score of approximately 84.2%.

## Macro F1

The project has 50 signs. We calculate F1 for every sign, then average those 50
scores with equal weight:

    macro F1 = average F1 across all signs

Macro F1 is the primary model-selection metric. It prevents a model from
appearing strong only because it works well on easy or frequent classes.

## Confusion Matrix

A confusion matrix records actual classes as rows and predicted classes as
columns. It identifies which signs are mixed up. For example, if many actual
thankyou samples are predicted as hello, we can inspect those videos and their
motion patterns.

## Correct experiment process

    689 train videos -> fit model
    77 validation videos -> choose model settings using macro F1
    192 test videos -> final evaluation only after choices are fixed

The test split must not be used to choose settings. This keeps its final score
an unbiased estimate of performance on unseen data.

## Models and metrics

| Model | Role | Selection metric |
|---|---|---|
| SVM (Support Vector Machine) | First classical baseline using summary features | Validation Macro F1 |
| GRU (Gated Recurrent Unit) | Future temporal sequence model | Validation Macro F1 |
| LSTM (Long Short-Term Memory) | Future temporal sequence model | Validation Macro F1 |
| Transformer | Future attention-based sequence model | Validation Macro F1 |

For every model, record validation accuracy, macro F1, per-class precision,
recall and F1, plus a confusion matrix.
