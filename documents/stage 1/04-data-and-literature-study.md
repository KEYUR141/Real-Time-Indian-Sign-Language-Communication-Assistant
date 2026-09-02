# Data and Literature Study

## 1. Data Strategy

The recommended strategy combines public resources with a controlled team-recorded dataset.

```text
Public ISL resources for study and baseline experiments
                         +
Team-recorded samples matching the final webcam environment
                         ->
Reviewed and versioned project dataset
```

Public data can accelerate research, but the team must verify licences, label definitions, signer splits, and recording quality before use. The final demonstration should be tested on data collected independently from training.

## 2. Candidate Resources

### INCLUDE

INCLUDE is a large-scale dataset for isolated Indian Sign Language recognition presented at ACM Multimedia 2020. It is relevant to fixed-vocabulary isolated recognition and can inform label selection and baseline experiments.

Reference: G. Sridhar et al., “INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition,” ACM Multimedia, 2020.

### iSign

iSign is an ISL processing benchmark that includes tasks involving sign videos, pose, text, and semantics. The ACL 2024 paper reports more than 118,000 video-sentence or phrase pairs. It is especially relevant to future pose-to-text and sentence-level research.

Reference: [A. Joshi et al., “iSign: A Benchmark for Indian Sign Language Processing,” Findings of ACL, 2024](https://aclanthology.org/2024.findings-acl.643/)

### ISLTranslate

ISLTranslate is a continuous ISL translation resource containing approximately 31,000 ISL-English sentence or phrase pairs. It is useful for understanding future translation work, but it is more complex than the first isolated-sign prototype.

Reference: [A. Joshi, S. Agrawal, and A. Modi, “ISLTranslate: Dataset for Translating Indian Sign Language,” 2023](https://arxiv.org/abs/2307.05440)

### ISLRTC Dictionary

The ISLRTC dictionary is an authoritative reference for verifying signs and variants. It should not automatically be treated as a balanced machine-learning dataset. Its stated research-use conditions require acknowledgement and prohibit resale or profiteering.

Reference: [ISLRTC FAQ](https://islrtc.nic.in/faq/)

### MediaPipe research

MediaPipe Hands provides real-time hand tracking using a palm detector and a hand-landmark model. The current Hand Landmarker returns 21 landmarks per detected hand, handedness, normalized coordinates, and world coordinates.

References:

- [MediaPipe Hands: On-device Real-time Hand Tracking](https://research.google/pubs/mediapipe-hands-on-device-real-time-hand-tracking/)
- [MediaPipe Hand Landmarker Result](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarkerResult)

## 3. Team Data Collection Target

### Pilot

- 3 signs
- 3 signers
- 20 accepted samples per sign per signer
- Target total: 180 accepted samples

### Expanded prototype

- 20 to 30 signs
- Preferably 8 to 10 signers
- Approximately 30 to 50 accepted samples per sign per signer, subject to available time

Signer diversity is more valuable than collecting many nearly identical recordings from one person.

## 4. Required Sample Metadata

Each sample should include:

```json
{
  "sample_id": "S03_water_012",
  "signer_id": "S03",
  "label": "water",
  "session_id": "2026-09-02-A",
  "handedness_note": "right-dominant",
  "camera_id": "laptop-webcam",
  "fps_target": 30,
  "quality_status": "approved",
  "consent_recorded": true
}
```

Personal names should not appear in filenames or model features.

## 5. Data Split Rule

Split by signer:

```text
Training signers   -> model learning
Validation signers -> model and threshold selection
Test signers       -> final evaluation only
```

Do not randomly distribute videos from the same signer across all three sets. That would measure familiarity with the signer rather than generalization to a new user.

## 6. Dataset Acceptance Checks

- Label matches the performed sign.
- Full relevant signing space is visible.
- Required hands and upper body are not cropped.
- The sample contains one intended sign.
- Recording starts before and ends after the sign.
- No second person creates false landmarks.
- Landmark-detection coverage is acceptable.
- Consent and participant code are present.
- Class and signer counts remain balanced.

## 7. Questions to Resolve in Stage 2

- Which three signs are visually distinct enough for the pilot?
- Which signs require body location or two hands?
- Are any selected signs differentiated mainly by facial expression?
- Which regional variants will be accepted?
- Can an ISL-knowledgeable reviewer verify the recording guide?
- What are the exact usage terms for every public dataset selected?

