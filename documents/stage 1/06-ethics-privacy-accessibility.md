# Ethics, Privacy, and Accessibility

## 1. Responsible Project Position

The project is an experimental communication aid for a limited vocabulary. It must not be presented as a replacement for ISL, Deaf culture, or qualified interpreters.

The system must not be relied on for high-stakes medical, legal, financial, emergency, or safety decisions.

## 2. Participant Consent

Before recording, each participant should understand:

- The purpose of the project
- What video, landmarks, and metadata will be stored
- Who can access the data
- How long the data will be kept
- Whether samples may appear in presentations
- That participation is voluntary
- How to request withdrawal before the agreed cutoff

Consent should distinguish between:

1. Use for model training and evaluation
2. Use of images or videos in reports and presentations
3. Future use beyond the current college project

Agreement to one purpose must not automatically imply agreement to every purpose.

## 3. Data Minimization

- Collect only information needed for the research question.
- Use participant codes instead of names in datasets.
- Store consent forms separately from model data.
- Do not store audio unless it is explicitly needed and consented to.
- Do not retain live inference video by default.
- Restrict copies of raw video.

Landmarks reduce visual detail but are still derived from a person and should be treated as sensitive research data.

## 4. Fairness and Representation

The model may perform differently across signers because of appearance, body proportions, signing style, disability, mobility, camera setup, and regional variation.

The team should:

- Include multiple signers where practical.
- Report signer counts and relevant collection conditions.
- Evaluate on unseen signers.
- Report per-class performance and failure cases.
- Avoid claiming that a small participant sample represents all ISL users.
- Record which sign variants are supported.

## 5. Failure and Uncertainty

The interface should communicate uncertainty honestly.

- Use an `unknown` or `not recognized` state.
- Do not display low-confidence guesses as facts.
- Let the user correct or remove output.
- Preserve the raw recognized label when optional sentence correction is used.
- Make it easy to stop the camera and speech.

## 6. Accessibility Requirements

- Provide visible status instead of relying on sound.
- Use captions and readable text.
- Maintain sufficient colour contrast.
- Do not use colour as the only error indicator.
- Support keyboard navigation for primary controls.
- Avoid very short timeouts for performing signs.
- Allow users to review output before speech playback.
- Avoid describing Deaf people as deficient or the system as “giving them a voice.”

## 7. Communication Guidelines

Preferred terms include:

- Deaf and hard-of-hearing people
- ISL users or signers
- Communication barrier
- Recognition assistance
- Supported vocabulary

Avoid terms that imply a person is mute, helpless, or needs to be corrected. The limitation belongs to the prototype, not to the language or its users.

## 8. Publication Checklist

Before showing results publicly:

- Confirm consent for every displayed recording or image.
- Remove personal names and unnecessary metadata.
- State vocabulary size and participant count.
- Explain that evaluation is signer-independent or disclose if it is not.
- State known limitations.
- Separate recognition results from LLM-generated corrections.
- Acknowledge datasets, ISLRTC resources, participants, and reviewers.

