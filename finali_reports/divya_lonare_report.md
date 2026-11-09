# Research and Documentation Report

**Name:** Divya Lonare

**Contribution area:** Research and Documentation

**Project:** Real-Time Indian Sign Language Communication Assistant

## Dataset Research

The project examined INCLUDE, Kaggle ISL, and ISLVT as potential data sources.
The reported model benchmarks used AI4Bharat's INCLUDE dataset. The other
downloaded sources were not included in those training and test results.

The initial INCLUDE-50 benchmark contained 958 videos across 50 classes.
The larger local benchmark contained 4,284 videos across 262 classes,
excluding eight unavailable videos belonging to the Second (Number) class.

## Technical Documentation

The project documentation explains the conversion of videos into MediaPipe
hand and upper-body landmarks, sequence sampling, normalization, feature
validation, and model evaluation.

It also describes the SVM, LSTM, and Compact Transformer approaches, their
benchmark results, and the distinction between dataset performance and
recognition of new webcam users.

## Report and Presentation Material

The final project report covers the problem statement, literature review,
dataset preparation, methodology, results, application testing, conclusions,
and future scope.

Presentation preparation material includes model terminology, explanations
of accuracy, precision, recall, and F1, the system flow, benchmark comparisons,
and the limitations of the current prototype.

## Documented Outcome

The documentation records the Compact Transformer as the selected model,
with 94.36% test accuracy and 94.47% test Macro F1 on the 262-class local
INCLUDE benchmark. It also records the remaining gap between dataset
recognition and prediction for new webcam users.
