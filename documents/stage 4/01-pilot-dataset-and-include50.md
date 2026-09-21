# Stage 4: Pilot Dataset and INCLUDE-50 Readiness

## 1. Purpose

Stage 4 tested the Stage 3 landmark pipeline on real INCLUDE video before
scaling to a multi-class benchmark. It then replaced the originally proposed
locally recorded 3-sign pilot with the official, reproducible INCLUDE-50
subset for preprocessing and model experiments.

## 2. Extraction Pilot

The first end-to-end extraction used the INCLUDE class `loud`:

| Metric | Result |
|---|---:|
| Videos selected | 21 |
| Successfully extracted | 21 |
| Failed | 0 |
| Valid landmark archives | 21 |
| Validation warnings | 0 |
| Pilot feature size | 1.109 MB |

This pilot exercised video decoding, temporal sampling, reusable MediaPipe
timestamps, hand and pose detection, normalization, compressed archive writing,
and independent archive validation.

These 21 pilot outputs were stored outside the official manifest-organized
paths. Later quality reporting correctly identified them as unexpected pilot
artifacts and excluded them from model training.

## 3. Dataset Used for the Modelling Milestone

Rather than claim completion of the proposed 180-video local participant
collection, the project adopted INCLUDE-50 as its controlled pilot benchmark.
It supplies official labels and fixed train, validation, and test manifests.

| Split | Classes | Samples | Local availability |
|---|---:|---:|---:|
| Train | 50 | 689 | 100% |
| Validation | 50 | 77 | 100% |
| Test | 50 | 192 | 100% |
| Total | 50 | 958 | 100% |

All 958 manifest-selected videos were converted to MediaPipe `.npz` archives.
Validation found 958 valid features, no missing features, no invalid features,
and no warnings. The 21 earlier `loud` pilot files remained explicitly
separate.

## 4. Quality Controls

The pilot and INCLUDE-50 work verified:

- exact manifest coverage by split and class;
- canonical labels, including paths containing nested `Extra` folders;
- collision-free feature paths using full manifest-relative identities;
- source metadata and fixed schema in every archive;
- valid sequence shapes, masks, and timestamps;
- per-class and per-split hand and pose coverage;
- resume detection for complete existing archives;
- atomic writes and checkpointed extraction reports.

## 5. Limitations

INCLUDE is an existing isolated-sign dataset, not a new participant collection
created by this project. Consequently, Stage 4 does not establish:

- informed-consent records for newly recruited local participants;
- a controlled 3-sign by 3-signer local recording balance;
- signer-independent performance unless signer metadata proves the split;
- generalization to live webcams or continuous signing.

Those limitations remain deployment and evaluation requirements rather than
being hidden by the successful archive validation.

## 6. Stage Decision

Stage 4 is accepted as a dataset-readiness milestone:

1. The 21-video `loud` pilot validated the extraction mechanics.
2. INCLUDE-50 supplied the reproducible multi-class dataset.
3. The 958 official features passed manifest-aware validation.
4. The dataset was approved for preprocessing and baseline modelling.

The resulting preprocessing quality and model benchmarks are documented in
Stages 5 and 6.

