# Stage 1 Completion Checklist

## 1. Domain Understanding

- [ ] The team can explain why ISL is a language rather than a collection of gestures.
- [ ] The team understands handshape, orientation, location, movement, and non-manual signals.
- [ ] The team can distinguish isolated recognition, continuous recognition, and translation.
- [ ] The team agrees not to describe the first release as unrestricted ISL translation.

## 2. Scope and Users

- [ ] Primary users and supporting users are agreed.
- [ ] The initial use case is approved.
- [ ] The in-scope and out-of-scope lists are approved.
- [ ] The project claim is used consistently in reports and presentations.
- [ ] The demonstration environment and hardware are identified.

## 3. Data

- [ ] Public dataset candidates have been reviewed.
- [ ] Usage licences or terms have been checked before downloading data.
- [ ] The team-recorded pilot target is accepted.
- [ ] Required metadata fields are agreed.
- [ ] The signer-independent split rule is accepted.
- [ ] An ISL-knowledgeable person or authoritative resource can verify selected signs.

## 4. Requirements

- [ ] Must-have functional requirements are approved.
- [ ] Evaluation metrics are approved.
- [ ] Latency will be measured rather than assumed.
- [ ] Unknown-input and duplicate-output behaviour are required.
- [ ] Accessibility requirements are included in interface planning.

## 5. Ethics and Privacy

- [ ] A participant information and consent form will be prepared before recording.
- [ ] Data-retention and access rules are agreed.
- [ ] Presentation consent is separate from training-data consent.
- [ ] Personal names will not be used in dataset filenames.
- [ ] The team understands that landmarks remain sensitive derived data.
- [ ] High-stakes use is explicitly excluded.

## 6. Technical Decisions

- [ ] MediaPipe is understood as a feature extractor, not the team-trained sign classifier.
- [ ] Both hands and selected upper-body pose features are the initial input.
- [ ] Full facial landmarks are deferred unless the vocabulary requires them.
- [ ] A simple baseline will be trained before the sequence model.
- [ ] Real-time acceptance will use confidence and temporal stability.
- [ ] The first implementation target is a three-sign vertical prototype.

## 7. Open Decisions for Stage 2

Record decisions here during the team review:

| Decision | Owner | Due date | Status |
|---|---|---|---|
| Select three pilot signs | AI lead and ISL reviewer | TBD | Open |
| Confirm accepted sign variants | ISL reviewer | TBD | Open |
| Confirm participant availability | Team coordinator | TBD | Open |
| Confirm demonstration laptop and webcam | Development lead | TBD | Open |
| Select model framework for sequence training | AI lead | Before Stage 8 | Deferred |

## 8. Stage Approval

Stage 1 is complete when all blocking checklist items are resolved.

| Role | Name | Review date | Approved |
|---|---|---|---|
| Project supervisor |  |  |  |
| AI lead |  |  |  |
| Team representative |  |  |  |
| ISL reviewer, if available |  |  |  |

