# Stage 2: Data Collection Report

## Collection status

**Status: complete for INCLUDE-50 and usable for project development.**

The AI4Bharat INCLUDE dataset was downloaded from its official Zenodo distribution and extracted to the external dataset drive. The official split manifests and label maps were downloaded from the AI4Bharat INCLUDE repository.

The project will begin with **INCLUDE-50**, which has been verified as complete. The full INCLUDE collection is also available, with one documented source/archive discrepancy affecting 8 of the 4,292 manifest entries.

## Dataset identity

| Field | Value |
|---|---|
| Dataset | INCLUDE: Indian Lexicon Sign Language Dataset |
| Provider | AI4Bharat, IIT Madras |
| Task | Isolated Indian Sign Language recognition |
| Dataset version | Zenodo v1 |
| Official release year | 2020 |
| Official dataset record | <https://zenodo.org/records/4010759> |
| Reference implementation | <https://github.com/AI4Bharat/INCLUDE> |
| Hugging Face metadata page | <https://huggingface.co/datasets/ai4bharat/INCLUDE> |
| Paper DOI | <https://doi.org/10.1145/3394171.3413528> |
| Dataset-card licence | CC BY 4.0 |

The dataset contains videos of individual signs rather than complete signed sentences. It is suitable for the project's initial isolated-sign recognition system, but it is not by itself a continuous ISL translation dataset.

## Local storage layout

The collected data is stored outside the Git repository at:

```text
E:\Hugging_Face\INCLUDE
├── raw\include\
│   ├── Adjectives\
│   ├── Animals\
│   ├── Clothes\
│   ├── Colours\
│   ├── Days_and_Time\
│   ├── Electronics\
│   ├── Greetings\
│   ├── Home\
│   ├── Jobs\
│   ├── Means_of_Transportation\
│   ├── People\
│   ├── Places\
│   ├── Pronouns\
│   ├── Seasons\
│   └── Society\
├── manifests\
│   ├── label_maps\
│   └── train_test_paths\
├── .state\include\
└── .downloads\include\
```

The raw video data is stored only once. INCLUDE-50 is defined by its label map and split manifests; it does not require a duplicate copy of the videos.

## Verified local inventory

| Metric | Verified value |
|---|---:|
| Extracted video files | 4,284 |
| Extracted video size | 53.84 GB |
| Empty video files | 0 |
| Top-level semantic categories | 15 |
| Official full INCLUDE classes | 263 |
| Locally populated full-dataset classes | 262 |
| INCLUDE-50 classes | 50 |
| Complete INCLUDE-50 videos | 958 |
| Free space on E: after collection and verification | 60.15 GB |

The official Zenodo distribution consists of 44 ZIP archives totaling approximately **56.8 GB compressed**. The download script processed archives sequentially, verified each available MD5 checksum before extraction, and deleted completed ZIP files to avoid retaining a second full copy of the dataset.

## Manifest verification

### INCLUDE-50

INCLUDE-50 is fully present and is the selected dataset for the first modelling milestone.

| Split | Manifest entries | Locally present | Missing | Completeness |
|---|---:|---:|---:|---:|
| Train | 689 | 689 | 0 | 100% |
| Validation | 77 | 77 | 0 | 100% |
| Test | 192 | 192 | 0 | 100% |
| **Total** | **958** | **958** | **0** | **100%** |

The original publication describes INCLUDE-50 as 766 training videos and 192 test videos. In the repository manifests, those 766 development videos are divided into **689 train + 77 validation**.

### Full INCLUDE

| Split | Manifest entries | Locally present | Missing | Completeness |
|---|---:|---:|---:|---:|
| Train | 3,127 | 3,121 | 6 | 99.81% |
| Validation | 348 | 347 | 1 | 99.71% |
| Test | 817 | 816 | 1 | 99.88% |
| **Total** | **4,292** | **4,284** | **8** | **99.81%** |

All eight missing manifest entries belong to one class:

```text
Days_and_Time/Second (Number)/
```

Missing paths:

```text
Train:
Days_and_Time/Second (Number)/MVI_4647.MOV
Days_and_Time/Second (Number)/MVI_4649.MOV
Days_and_Time/Second (Number)/MVI_5503.MOV
Days_and_Time/Second (Number)/MVI_5504.MOV
Days_and_Time/Second (Number)/MVI_5505.MOV
Days_and_Time/Second (Number)/MVI_5506.MOV

Validation:
Days_and_Time/Second (Number)/MVI_4648.MOV

Test:
Days_and_Time/Second (Number)/MVI_4650.MOV
```

This discrepancy does not affect INCLUDE-50. If the project later trains on full INCLUDE, it must either recover these eight videos or explicitly exclude the `Second (Number)` class and its manifest entries.

## Verification notes

The following checks were completed:

- All 44 official dataset archives were processed.
- Archive checksums were verified before extraction by the download script.
- The extracted dataset contains no zero-byte video files.
- Manifest entries were checked using their complete category/class-relative paths.
- INCLUDE-50 has no missing train, validation, or test entries.
- Full INCLUDE has exactly eight missing entries, all isolated to one class.
- The `.MOV` versus `.MP4` extension difference in part of the extracted data was identified.

Some manifest entries end in `.MOV`, while the corresponding extracted files end in `.MP4`. Verification therefore compared the complete relative path while normalizing the extension. The data loader must apply the same resolution rule rather than assume the manifest extension always matches the extracted file extension.

Repeated basenames such as `MVI_####` can occur in different semantic folders. Basename repetition is not evidence of duplicated samples; dataset paths must always include category and class directories.

## What the collected data supports

The collected data supports:

- classification of one isolated ISL sign per video;
- development with a manageable 50-sign vocabulary;
- later expansion toward 262 locally populated classes from full INCLUDE;
- RGB video modelling;
- MediaPipe hand, pose, and face landmark extraction;
- comparison of video-based and landmark-based recognition models;
- evaluation with the official train, validation, and test manifests.

It does not independently support:

- unrestricted ISL conversation recognition;
- automatic segmentation of continuous signing;
- complete ISL sentence translation;
- national coverage of all regional ISL variants;
- claims that performance will generalize to every signer or webcam environment.

The source data was recorded in Chennai, Tamil Nadu. Regional and deployment-specific evaluation will therefore remain necessary.

## Stage 2 decision

Stage 2 data collection is accepted with the following decision:

1. Use **INCLUDE-50** for the first MediaPipe and recognition pipeline.
2. Preserve the official train, validation, and test manifests.
3. Keep raw videos immutable; write landmarks and processed outputs to separate directories.
4. Normalize `.MOV` and `.MP4` path differences in the data loader.
5. Treat the eight missing full-INCLUDE entries as a documented limitation.
6. Revisit the missing `Second (Number)` class only when expanding beyond INCLUDE-50.

## Presentation-ready figures

The following figures can be used directly in the project presentation:

- **4,284 ISL videos collected and locally verified**
- **53.84 GB of extracted sign-video data**
- **15 semantic categories**
- **263 official INCLUDE sign classes**
- **50 classes selected for the initial recognition milestone**
- **958/958 INCLUDE-50 videos available—100% complete**
- **689 training, 77 validation, and 192 test videos in INCLUDE-50**
- **0 empty video files**
- **4,284/4,292 full-INCLUDE manifest videos available—99.81% complete**
- **8 unavailable videos isolated to one non-INCLUDE-50 class**

Suggested slide statement:

> We collected and verified 4,284 isolated Indian Sign Language videos across 15 semantic categories. For the initial model, we selected the complete INCLUDE-50 benchmark: 958 videos covering 50 signs, with official train, validation, and test splits. This provides a controlled foundation for MediaPipe-based landmark extraction and real-time isolated-sign recognition.

## Stage 2 completion checklist

- [x] Primary ISL video source selected.
- [x] Official dataset archives downloaded.
- [x] Archives checksum-verified and extracted.
- [x] Official label maps downloaded.
- [x] Official split manifests downloaded.
- [x] Local video inventory measured.
- [x] Empty-file check completed.
- [x] INCLUDE-50 manifest coverage verified at 100%.
- [x] Full INCLUDE discrepancy documented.
- [x] Dataset stored outside Git on the E: data drive.
- [x] Dataset limitations and supported claims documented.
- [x] Dataset ready for Stage 3 MediaPipe processing.

