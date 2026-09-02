# Requirements and Success Criteria

## 1. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | The system shall capture video from a selected webcam. | Must |
| FR-02 | The system shall detect up to two hands. | Must |
| FR-03 | The system shall extract selected upper-body pose landmarks. | Must |
| FR-04 | The collection tool shall save the sign label, signer ID, sample ID, and recording metadata. | Must |
| FR-05 | The preprocessing pipeline shall produce repeatable fixed-schema features. | Must |
| FR-06 | The model shall classify only signs in the supported vocabulary. | Must |
| FR-07 | The inference pipeline shall expose prediction confidence. | Must |
| FR-08 | The system shall avoid emitting a label when confidence or stability is insufficient. | Must |
| FR-09 | The interface shall show recognized text and recognition status. | Must |
| FR-10 | The interface shall allow a user to correct, clear, and stop output. | Must |
| FR-11 | The interface shall optionally speak accepted text. | Should |
| FR-12 | The system shall maintain session and error logs without silently storing raw video. | Should |

## 2. Non-Functional Requirements

### Accuracy

The model must be evaluated on signers who do not appear in the training set. Overall accuracy alone is insufficient. The report must include macro F1, per-class recall, and a confusion matrix.

### Latency

The final system should feel interactive. A preliminary target is to display an accepted result within approximately one second after the sign sequence is available. Actual latency must be measured and reported.

### Reliability

- Missing hands must not crash the application.
- Unsupported input must not be forced into a known class.
- Repeated frames must not continuously duplicate the same word.
- The model, label map, and preprocessing version must match.

### Reproducibility

- Dataset versions must be identifiable.
- Train, validation, and test signer lists must be stored.
- Model parameters and random seeds must be recorded.
- Training and evaluation must be runnable from documented commands.

### Usability and accessibility

- Main controls must be keyboard accessible.
- Recognition state must not be communicated by colour alone.
- Text must be readable at presentation distance.
- Users must be able to stop the camera and speech output.
- The interface must show the supported vocabulary and limitations.

### Privacy and security

- Camera access requires explicit permission.
- Raw video must not be retained during normal inference unless the user has agreed.
- Participant identifiers should be coded rather than named in model files.
- Data access should be limited to the project team.

## 3. Pilot Success Criteria

The three-sign pilot is successful when:

- At least three consenting signers are represented.
- The complete capture-to-prediction pipeline runs without manual feature editing.
- Test recordings belong to an unseen signer.
- The system can output `unknown` or withhold output for unsuitable input.
- Major confusions and failure cases are documented.
- Real-time inference works on the intended demonstration laptop.

No fixed accuracy target should be promised before inspecting the pilot data. After the baseline is measured, the team should set a justified target for the expanded vocabulary.

## 4. Final Evaluation Questions

1. How accurately does the system recognize each supported sign for unseen signers?
2. Which signs are most frequently confused and why?
3. How often does the system produce a false output when no supported sign is performed?
4. What is the end-to-end prediction latency?
5. How does performance change with lighting, distance, background, and signing speed?
6. Can a first-time user understand camera placement, system status, and corrections?

