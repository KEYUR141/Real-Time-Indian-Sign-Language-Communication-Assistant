# Problem, Users, and Scope

## 1. Problem Statement

ISL users can face communication barriers when interacting with people who do not understand ISL. Human interpreters may not be available in every low-risk, routine interaction. Existing student systems often recognize only alphabets or carefully posed static gestures, while meaningful signs may depend on both hands, upper-body location, and movement.

The project will investigate whether a standard webcam, MediaPipe landmarks, and a team-trained sequence classifier can recognize a limited vocabulary of isolated ISL signs in near real time and present the result as text with optional speech.

## 2. Intended Users

### Primary users

- ISL users participating in a controlled demonstration
- Non-signers receiving recognized text or speech

### Supporting users

- Team members collecting and reviewing data
- Project supervisor evaluating the system
- Test participants providing usability feedback

## 3. Initial Use Case

1. A user opens the web application.
2. The user grants camera permission and starts a recognition session.
3. The user performs one supported sign within the camera frame.
4. The system extracts hand and pose landmarks.
5. The model predicts a sign and provides a confidence score.
6. Stability rules accept the result or show that the sign is unknown.
7. The accepted result appears as text.
8. The user may play the text through text-to-speech, correct it, or clear it.

## 4. Initial Scope

The first usable release will include:

- A controlled vocabulary of approximately 20 to 30 isolated ISL signs
- A three-sign pilot before vocabulary expansion
- Static and selected dynamic signs
- One signer visible at a time
- A standard RGB webcam
- Both-hand and selected upper-body landmark extraction
- Landmark-sequence classification
- Real-time text output
- Optional text-to-speech
- Confidence, stability, duplicate-suppression, and unknown handling
- Evaluation on participants who were not used for training

## 5. Out of Scope for the First Release

- Unrestricted continuous ISL conversation
- Complete ISL-to-English or ISL-to-Hindi translation
- Automatic recognition of every ISL dictionary entry
- Interpretation of complex grammar and discourse
- Guaranteed operation in medical, legal, or emergency decisions
- Multiple simultaneous signers
- Mobile-device optimization
- Replacement of qualified human interpreters
- Automatic LLM correction presented as if it were the original recognition

## 6. Assumptions

- The user is aware of the supported vocabulary.
- The upper body and both hands can be visible when required.
- Participants provide informed consent for recording and storage.
- The team can access a normal laptop and webcam.
- The selected signs can be verified through a reliable ISL source.
- Real-time means an interactive response, not guaranteed zero latency.

## 7. Constraints

- Small college-team capacity
- Limited access to fluent ISL signers
- Limited GPU resources
- Time required for ethical data collection
- Landmark failure during occlusion or poor lighting
- Public datasets may use different labels and recording conditions
- A small vocabulary limits real-world usefulness

## 8. Project Claim

Use this wording consistently:

> A real-time communication assistant that recognizes a selected vocabulary of isolated Indian Sign Language signs using webcam-based landmark sequences and converts accepted predictions into text and optional speech.

