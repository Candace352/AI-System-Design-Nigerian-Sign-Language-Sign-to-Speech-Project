# Nigerian Sign Language Sign-to-Speech Using YOLO26

## 1. Abstract (238 words)

This project proposes a low-cost, real-time Nigerian Sign Language (NSL) sign-to-speech prototype that helps reduce communication barriers between Deaf or hard-of-hearing signers and people who do not understand NSL. Building on the 2021 NSL case study, which trained YOLOv5m on labelled still images, the project will fine-tune a modern YOLO26 object-detection model on the publicly available NSL dataset. Each camera frame will be analysed to locate a performed sign and assign one of the available word, phrase, or alphabet labels. A reliability layer will speak a label only when its confidence is sufficiently high and it remains stable across consecutive frames; uncertain predictions will be displayed as a request to repeat rather than spoken. The initial scope deliberately supports one static, isolated sign at a time; it does not claim to translate full NSL sentences, whose meaning can depend on motion, facial expression, body posture, and context.

The main experiment will compare YOLO26n and YOLO26s fine-tuning runs and use the paper's reported YOLOv5m figures as a historical baseline. A signer-disjoint evaluation will be used if the metadata permits; otherwise, a small cross-condition test with unseen users will be proposed. Success will be judged by detection quality, speed, and the reliability layer's ability to reduce incorrect spoken outputs. The intended outcome is a reproducible system-design prototype and an evidence-based account of whether a newer YOLO version improves the accuracy–latency trade-off for this low-resource African sign-language task.

## 2. Introduction and SDG relevance

People who use NSL can face communication barriers when an interpreter is unavailable or when the listener does not know NSL. The supplied study demonstrated an initial sign-to-text-to-speech prototype, but it used YOLOv5m and a small, static-image dataset. This project updates the detection component to YOLO26 and evaluates it carefully in a practical webcam workflow.

The project supports inclusion: an accessible, on-device-capable interface can make short everyday interactions more independent. It must be designed *with* NSL users, however; the tool assists communication and must not be presented as a replacement for human interpreters or as a universal translator.

## 3. Selected SDG

**SDG 10 — Reduced Inequalities**, especially **Target 10.2**: promote the social, economic, and political inclusion of all people, including persons with disabilities. SDG 4 (Quality Education) is a secondary benefit because accessible communication can support learning, but SDG 10 is the primary focus.

## 4. Refined problem statement

**How can a lightweight webcam application use YOLO26 to recognise a bounded vocabulary of isolated Nigerian Sign Language signs and produce reliable real-time speech across unseen signers or acquisition conditions?**

## 5. AI task formulation

**Input:** a 640 × 640 RGB camera frame containing one signer performing one of the supported static signs.

**Model task:** supervised multi-class object detection. YOLO26 predicts bounding boxes, class probabilities, and confidence scores for the sign region.

**Output:** the highest-confidence label, bounding box, confidence, on-screen text, and text-to-speech audio only after a high-confidence label is stable for several frames. Otherwise, the system abstains and asks the user to repeat the sign.

**Out of scope for this first version:** continuous signing, grammatical translation, signs whose identity depends primarily on motion, multi-person conversations, and interpreting facial grammar. Those require video/pose/sequence models and more representative data.

## 6. Initial data sources

1. [Lanfrica / Hugging Face NSL dataset](https://huggingface.co/datasets/Lanfrica/sign-to-speech-for-sign-language-understanding-a-case-study-of-nigerian-sign-language) — the public dataset linked to the case study (about 5,000 cleaned images, 137 labels; check its licence and access conditions before use).
2. [Original project repository](https://github.com/ML-Collective/Sign-to-Speech-for-Sign-Language-Understanding) — original code, configurations, and dataset information.
3. **Optional local extension:** a small, consent-based collection of NSL users in the intended deployment setting. Capture different signers, lighting, camera angles, skin tones, clothing, and backgrounds. Obtain informed consent, minimise personal data, and do not upload identifiable videos without permission.

The public data should be inspected before training. Images from the same person must not be randomly split between train and test; that would overstate performance by allowing the model to recognise the signer rather than the sign.

## 7. Expected challenges

- The dataset is small for 137 labels and may have few examples per class.
- A still image misses motion, facial expression, and body posture that can distinguish signs.
- Signer, lighting, background, camera, occlusion, and hand-size variation can reduce performance.
- Class imbalance, inaccurate boxes, and near-duplicate frames can bias results.
- A random image-level split can cause signer leakage; use a signer-disjoint split whenever possible.
- Incorrect spoken output can confuse or embarrass users. Show confidence, allow repeat/mute, and co-design the interface with NSL users.
- YOLO26 model size affects laptop/phone speed; evaluate `n` and `s` models before choosing a deployment model.

## 8. Preliminary success metrics

| Area | Metric | Initial target |
|---|---|---|
| Detection | mAP50-95 on a held-out signer-disjoint test set, where available | Report honestly; use the paper's 0.64 at IoU 0.95 as a historical reference, not a directly comparable result unless the split is reproduced |
| Recognition | Macro F1 and per-class recall | ≥ 0.80 macro F1 for a smaller, well-supported initial vocabulary; report all 137-class result separately |
| Reliability | False speech events per 5-minute session | ≤ 2 after temporal stabilisation |
| Reliability | Incorrect spoken outputs per 100 signing events, before vs. after the reliability layer | Demonstrate a reduction after confidence filtering and temporal stabilisation |
| Reliability | Correct/incorrect abstentions per signing event | Report the trade-off: fewer incorrect outputs may cause more abstentions |
| Speed | End-to-end webcam latency and FPS | ≥ 15 FPS on the target laptop; report hardware |
| Inclusion | Usability rating from 5+ NSL users | Median ≥ 4/5, plus qualitative feedback |
| Fairness | Gap in macro F1 across signers / lighting groups | ≤ 10 percentage points where sample sizes permit |

## Deployment resources and timeline

| Week | Work | Resources |
|---|---|---|
| 1 | Download, audit, licence-check, and organise labels; verify whether signer metadata exists | Dataset, annotation viewer |
| 2 | Create signer-disjoint splits if possible; otherwise design a small unseen-user test; train YOLO26n | Laptop/Colab GPU, Ultralytics |
| 3 | Train YOLO26s; compare accuracy, model size, FPS, and latency | GPU, experiment log |
| 4 | Build confidence + temporal-stability layer and webcam speech prototype; compare raw vs filtered outputs | Webcam, speakers, Python TTS |
| 5 | Evaluate recognition events, document limitations, and finalise report/demo | Test protocol, optional consent form |

## References

- Kolawole, S., Osakuade, O., Saxena, N., & Olorisade, B. K. (2021). *Sign-to-Speech Model for Sign Language Understanding: A Case Study of Nigerian Sign Language.* [arXiv:2111.00995](https://arxiv.org/abs/2111.00995).
- [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/).
- [United Nations SDG 10](https://sdgs.un.org/goals/goal10).
