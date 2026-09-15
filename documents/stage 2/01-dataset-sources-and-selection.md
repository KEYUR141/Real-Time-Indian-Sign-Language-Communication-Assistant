# Stage 2: Dataset Sources and Selection

## Purpose

This document defines how datasets will be discovered, assessed, selected, and combined for the **Real-Time Indian Sign Language (ISL) Communication Assistant**. The Stage 2 outcome is a traceable dataset decision—not merely a download—covering suitability, licensing, label quality, signer diversity, technical compatibility, and risks.

## Project requirements

The target system must recognize Indian Sign Language from a live camera feed with sufficiently low latency for interactive communication. Dataset selection must therefore support the intended recognition scope and deployment conditions.

Initial assumptions:

- Input is RGB video from an ordinary webcam or phone-class camera.
- The first production milestone should use a bounded vocabulary rather than claim unrestricted ISL translation.
- Both manual features (hands, motion, orientation, location) and non-manual features (face and upper-body cues) may carry meaning.
- Training, validation, and test data must be separated by signer wherever identities are available.
- Static alphabet or digit datasets can support prototyping, but cannot by themselves validate continuous sign recognition.
- ISL must not be treated as a word-for-word encoding of English or Hindi.

## Dataset categories

Candidate data is grouped by the capability it can support.

| Category | Typical content | Appropriate use | Main limitation |
|---|---|---|---|
| Static signs | Isolated alphabet, digits, or poses in images | Camera pipeline, hand detection, baseline classifier | Does not represent motion or continuous signing |
| Isolated dynamic signs | One labelled sign per video clip | Initial vocabulary recognizer and real-time demo | Limited coarticulation and sentence context |
| Continuous signing | Sentences or discourse with temporal annotations | Segmentation, sequence recognition, translation | Scarce, expensive to annotate, and harder to evaluate |
| Lexical/reference resources | Dictionary entries, glosses, example videos | Vocabulary definition and label normalization | Often unsuitable for bulk model training or redistribution |
| Supplemental non-ISL data | Generic hand, pose, or face data | Pretraining detectors and representation learning | Must not be reported as ISL recognition data |

## Candidate source families

The following source families should be investigated. A source is not approved until its current access terms and licence are recorded in the dataset register.

### 1. Indian Sign Language Research and Training Centre resources

ISLRTC is the primary institutional reference for standardized ISL vocabulary and dictionary material. Its resources are valuable for vocabulary definition, gloss spelling, semantic review, and checking whether a proposed label represents an ISL sign.

Use as:

- a vocabulary and terminology reference;
- a source for human review by ISL-competent annotators;
- training data only if the applicable terms expressly permit downloading, derivative processing, and the intended project use.

Do not assume that publicly viewable dictionary media is automatically licensed for bulk training or redistribution.

### 2. Government, academic, and research datasets

Search repositories and accompanying papers from Indian universities, research laboratories, and government initiatives for isolated and continuous ISL corpora. Prefer datasets that provide:

- an authoritative project or institutional page;
- a paper describing collection and annotation;
- explicit licence or written permission;
- stable download access and version information;
- signer-disjoint evaluation guidance;
- metadata for signers, capture setup, and labels.

A paper mentioning a dataset is not sufficient if the data is unavailable or the usage rights are unclear.

### 3. Public dataset platforms

Kaggle, GitHub, Hugging Face, Zenodo, Mendeley Data, IEEE DataPort, and similar platforms may contain useful ISL datasets or mirrors. Treat the platform as hosting, not as proof of provenance.

Before selection, trace each candidate to its original creator and confirm:

- the uploader is authorized to distribute it;
- the licence applies to the media, not only to accompanying code;
- the source has not relabelled ASL or generic gestures as ISL;
- duplicate or extracted dictionary videos are not presented as independently collected data;
- consent and privacy information are adequate for videos containing identifiable people.

### 4. Project-collected data

If no existing source meets the required vocabulary, diversity, or licence constraints, collect a focused project dataset. Collection should use informed consent, a written data-use policy, controlled label prompts, and review by an ISL expert or fluent signer.

Project collection is especially appropriate for:

- commands or phrases specific to the application;
- evaluation under the actual webcam and background conditions;
- filling signer, lighting, viewpoint, clothing, or device gaps;
- resolving label variants found during source normalization.

## Mandatory selection criteria

Each candidate is scored from 0 to 5 against the criteria below. A score of 0 on licence/permission, provenance, or ISL validity disqualifies the source regardless of its total.

| Criterion | Weight | What to verify |
|---|---:|---|
| ISL validity and label accuracy | 20% | Signs are genuinely ISL; labels and variants have competent review |
| Licence, consent, and permitted use | 20% | Training, local storage, derived features, publication, and redistribution rights are explicit |
| Match to target task | 15% | Static, isolated, or continuous format matches the planned milestone |
| Signer diversity | 10% | Multiple signers; useful age, gender, regional, appearance, and signing-style coverage |
| Visual and motion quality | 10% | Adequate resolution/frame rate; hands, face, and upper body remain visible |
| Annotation quality | 10% | Stable gloss vocabulary, temporal boundaries where needed, documented annotator process |
| Evaluation integrity | 10% | Signer identities or grouping keys allow leakage-free splits |
| Accessibility and reproducibility | 5% | Stable version, checksums, documentation, and repeatable acquisition |

Weighted score:

```text
score = sum((criterion_score / 5) * criterion_weight)
```

Interpretation:

- **80–100:** strong primary candidate;
- **65–79:** usable with documented mitigation;
- **50–64:** prototype or supplemental use only;
- **below 50:** reject;
- **any mandatory zero:** reject.

## Selection workflow

1. Define the Stage 2 vocabulary and whether the milestone is static, isolated dynamic, or continuous recognition.
2. Create one register entry per candidate using the template below.
3. Verify the original source, version, licence, consent statement, and any access restrictions.
4. Download a small permitted sample and inspect labels, video properties, signer distribution, and duplicates.
5. Ask an ISL-competent reviewer to validate a stratified sample of signs and variants.
6. Score the source using the weighted rubric.
7. Test whether signer-disjoint splits are possible before committing to model development.
8. Select one primary dataset and only the minimum supplemental data needed to close measured gaps.
9. Freeze a versioned manifest with file hashes; preserve raw data unchanged and generate processed data separately.
10. Record the final decision, exclusions, risks, and licence obligations.

## Dataset register template

Maintain the register in a machine-readable file in a later implementation step; use the following fields:

| Field | Required content |
|---|---|
| Dataset ID | Short stable project identifier |
| Name and version | Creator-provided name and release/version date |
| Original URL | Primary institutional, repository, or DOI link |
| Citation | Paper or documentation citation |
| Owner/creator | Institution and responsible contact, if available |
| Licence | Exact licence name and link or permission record |
| Permitted uses | Training, research, commercial use, derivatives, redistribution |
| Modality | Images, isolated videos, continuous videos, keypoints, or mixed |
| Vocabulary | Number of glosses/classes and language coverage |
| Samples | Total items and approximate duration/frame count |
| Signers | Count and available demographic/region metadata |
| Capture conditions | Resolution, FPS, viewpoint, background, devices |
| Annotation | Glosses, translations, temporal boundaries, confidence/review method |
| Split support | Signer/session/group identifiers available for splitting |
| Known issues | Missing files, imbalance, duplicates, ambiguity, quality problems |
| Score and decision | Weighted score plus accept/conditional/reject rationale |
| Integrity | Archive/file checksums and acquisition date |

## Data audit after acquisition

Run the following checks before preprocessing:

- Confirm file counts, readable media, duration, resolution, frame rate, and codec distribution.
- Calculate class counts and signer-by-class coverage.
- Detect exact duplicates with cryptographic hashes and probable near-duplicates with perceptual/video fingerprints.
- Find label spelling differences, synonyms, homonyms, numeric IDs, and language mixing.
- Identify clips with cropped hands, missing facial cues, severe blur, poor lighting, occlusion, or multiple active signers.
- Check whether backgrounds, clothing, recording sessions, or watermarks are correlated with labels.
- Confirm that the same signer, session, or source clip cannot cross evaluation boundaries.
- Manually inspect a stratified sample, including rare classes and every proposed evaluation class.

## Label normalization

Create a canonical label map without overwriting source labels. Each row should preserve:

- original dataset and original label;
- canonical gloss used by this project;
- human-language meaning or short description;
- static/dynamic indicator;
- one-hand/two-hand indicator where useful;
- known regional or lexical variant;
- reviewer and review status;
- ambiguity or exclusion reason.

Labels from separate datasets must not be merged solely because their English translations match. Merge only after sign-form and meaning validation.

## Split and leakage policy

The preferred split is signer-disjoint:

- training: approximately 70–80% of signers;
- validation: approximately 10–15% of signers;
- test: approximately 10–20% of signers.

Exact ratios may change for small datasets, but all samples from one signer—and preferably one recording session—must remain in one split. If signer IDs are unavailable, the dataset cannot support a trustworthy generalization claim; it may be used for pipeline prototyping only.

Deduplicate before splitting. When combining sources, treat the original video or dictionary entry as the grouping unit so derived clips and mirrors remain together.

## Recommended Stage 2 decision

Use a staged data strategy:

1. **Primary milestone:** choose a licensed, multi-signer isolated-video ISL dataset for a bounded vocabulary and signer-independent evaluation.
2. **Vocabulary authority:** use ISLRTC or another recognized ISL reference to normalize and review labels, subject to its usage terms.
3. **Deployment evaluation:** collect a small, consented dataset under the intended webcam conditions using signers excluded from training.
4. **Later expansion:** add continuous-sign data only after the isolated-sign baseline, evaluation protocol, and latency target are stable.

Static alphabet images may be used to validate the interface and preprocessing pipeline, but success on them must not be presented as a real-time ISL communication solution.

## Acceptance gates

The selected primary dataset must satisfy all of the following before Stage 2 closes:

- [ ] Original provenance and version are recorded.
- [ ] Licence or written permission covers the planned use.
- [ ] Consent/privacy conditions are documented.
- [ ] A competent reviewer confirms that the sampled labels represent ISL.
- [ ] The data supports the chosen recognition task and target vocabulary.
- [ ] Multiple signers are represented.
- [ ] Signer-disjoint validation and test splits can be created.
- [ ] Duplicate, imbalance, and media-quality audits are complete.
- [ ] Canonical labels preserve source labels and variants.
- [ ] Raw files and the acquisition manifest are immutable and checksummed.
- [ ] Dataset limitations are reflected in project claims and evaluation metrics.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| ASL or generic gestures are mislabeled as ISL | Verify provenance and require sample review by an ISL-competent person |
| Dictionary media has unclear reuse rights | Use it only as reference until explicit permission is confirmed |
| Model memorizes signers or recording backgrounds | Use signer/session-disjoint splits and inspect label-correlated artifacts |
| Vocabulary labels hide regional variants | Preserve variants in metadata and report supported coverage narrowly |
| Dataset is dominated by static signs | Keep static experiments as prototypes; select motion video for the main milestone |
| Class imbalance inflates overall accuracy | Report macro metrics, per-class recall, confusion matrix, and balanced sampling results |
| Combining sources creates domain shortcuts | Track source IDs, test per source, and include cross-source evaluation |
| Faces and bodies create privacy obligations | Minimize access, document consent, avoid public redistribution unless allowed |

## Stage 2 deliverables

- Completed candidate dataset register with verified links and licensing notes.
- Weighted comparison matrix and written selection decision.
- Immutable acquisition manifest containing versions and checksums.
- Dataset audit report covering labels, signers, balance, duplication, and media quality.
- Canonical label map with ISL review status.
- Reproducible signer-disjoint split manifests.
- Data card describing intended use, limitations, ethics, and excluded claims.

## Decision record

The final source names and URLs are intentionally left pending verification. Populate this section only after the current licence, availability, provenance, and sample quality have been checked.

| Role | Selected dataset/source | Status | Rationale |
|---|---|---|---|
| Primary training dataset | To be verified | Pending | Must pass all mandatory gates and support isolated dynamic ISL |
| Vocabulary reference | ISLRTC or equivalent authoritative ISL resource | Conditional | Use depends on the applicable access and reuse terms |
| Deployment test set | Project-collected, consented webcam recordings | Planned | Measures performance in the real target environment |
| Supplemental pretraining | Generic hand/pose data if required | Optional | May improve feature extraction but is not ISL evidence |

