# ISL Domain Study

## 1. What Is Indian Sign Language?

Indian Sign Language is a natural visual language used by members of the Deaf and hard-of-hearing community in India. Meaning is communicated through coordinated manual and non-manual signals rather than by replacing each spoken word with a hand gesture.

ISL has its own linguistic structure. It should not be treated as signed English, signed Hindi, mime, or a universal sign language. American Sign Language and ISL are also different languages.

## 2. Components of a Sign

A recognition system may need to observe several components:

1. **Handshape:** The configuration of the fingers and palm.
2. **Orientation:** The direction in which the palm and fingers face.
3. **Location:** The position of the sign relative to the face, chest, body, or signing space.
4. **Movement:** The direction, speed, repetition, path, and manner of motion.
5. **Two-hand relationship:** Whether one or both hands are active and how they interact.
6. **Non-manual signals:** Facial expression, mouth pattern, head movement, gaze, and body posture.

Two signs may look similar in a single frame but have different movement or location. This is why a sequence of frames is necessary for dynamic signs.

## 3. Recognition and Translation Are Different

### Isolated sign recognition

The user performs one sign at a time. The system predicts one label from a fixed vocabulary.

```text
One recorded sign -> Class label such as WATER
```

This is the scope of the initial project.

### Continuous sign recognition

The user signs a sequence without manually separating every sign. The system must detect sign boundaries and identify the sequence of signs.

### Sign-language translation

The system converts a signed utterance into a natural spoken-language sentence. Translation involves recognition, linguistic context, ISL grammar, and reordering. It is considerably more difficult than isolated recognition.

The first release must be described as an **ISL communication assistant for selected isolated signs**, not as a complete universal translator.

## 4. Variation in Real Use

Recognition is affected by:

- Regional and community sign variants
- Individual signing style and speed
- Left-handed and right-handed signing
- Hand size, body proportions, clothing, and skin appearance
- Camera position, lighting, background, and motion blur
- Partial hand occlusion and two-hand overlap
- Differences between learners and fluent signers

The project therefore needs multiple signers and signer-independent evaluation.

## 5. Role of the Deaf Community

Sign vocabulary and recording instructions should be reviewed by a qualified ISL user, instructor, interpreter, or the authoritative ISLRTC dictionary. The technical team should not invent signs or infer ISL signs from ASL resources.

Community participation improves linguistic correctness, usability, and respectful representation. Participants must be treated as contributors, not merely as data sources.

## 6. Domain Implications for the System

| Domain fact | Technical implication |
|---|---|
| Handshape matters | Capture detailed landmarks for both hands |
| Location matters | Retain selected upper-body pose landmarks |
| Motion matters | Use ordered frame sequences for dynamic signs |
| Facial cues may matter | Add selected face features only when required by the vocabulary |
| Signs vary between people | Train and test with different signers |
| Similar signs can be ambiguous | Use confidence thresholds and an `unknown` class or state |
| Signing is a language | Avoid claiming word-for-word translation from a small classifier |

## 7. Authoritative Starting Resource

The Indian Sign Language Research and Training Centre provides an ISL dictionary with sign videos and English or Hindi equivalents. Its FAQ states that the dictionary contains 10,000 terms and may be used for research or technology development with acknowledgement and without resale or profiteering.

Reference: [ISLRTC FAQ and Dictionary Information](https://islrtc.nic.in/faq/)

