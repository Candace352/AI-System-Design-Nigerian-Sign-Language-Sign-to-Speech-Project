# Nigerian Sign Language Sign-to-Speech Using YOLO26

## 1. Abstract (190 words)

This project develops a real-time Nigerian Sign Language (NSL) sign-to-speech prototype to reduce communication barriers between NSL users and people who do not understand NSL. It builds on Kolawole et al.'s 2021 study, which used YOLOv5m to detect isolated NSL signs from still images and converted predicted labels into speech. The proposed system will fine-tune YOLO26, a newer real-time object-detection model, on the public NSL dataset. A webcam frame will be processed to detect a supported sign, display its label, and generate speech. To reduce unreliable output, the system will include a confidence and temporal-stability layer: it will speak only when a high-confidence prediction remains consistent across consecutive frames, and otherwise request the user to repeat the sign.

The work contributes more than a model upgrade in two ways. First, it will evaluate generalisation with signer-disjoint data splits where metadata permits, or with a small cross-condition test involving unseen users otherwise. Second, it will compare raw YOLO predictions with reliability-aware spoken output, measuring whether filtering reduces incorrect speech at an acceptable latency cost. The scope is deliberately limited to isolated, static signs; it does not claim continuous NSL translation, which requires motion, facial expression, and linguistic context.

## 2. Introduction & SDG relevance

NSL users may face communication barriers when an interpreter is unavailable or when the listener does not understand NSL. Although the original study demonstrated a promising sign-to-speech system, it used YOLOv5m and a limited still-image dataset. This project modernises the detector with YOLO26 and focuses on dependable real-time output. It supports accessible communication but is an assistive prototype, not a replacement for NSL interpreters.

## 3. Selected SDG

**SDG 10: Reduced Inequalities**, particularly Target 10.2: promoting social inclusion for all people, including persons with disabilities.

## 4. Refined problem statement

How can YOLO26 support reliable, real-time recognition and speech output for a bounded vocabulary of isolated Nigerian Sign Language signs across unseen signers or acquisition conditions?

## 5. AI task formulation

**Input:** A webcam frame containing one supported static NSL sign.  
**Task:** Supervised multi-class object detection: YOLO26 predicts a bounding box, sign label, and confidence score.  
**Output:** Text and speech for a high-confidence label stable across several frames; otherwise, the system abstains and prompts the user to repeat the sign.  
**Out of scope:** Continuous signing, sentence translation, facial grammar, and signs differentiated mainly by motion.

## 6. Initial data sources

1. [Lanfrica/Hugging Face NSL dataset](https://huggingface.co/datasets/Lanfrica/sign-to-speech-for-sign-language-understanding-a-case-study-of-nigerian-sign-language): about 5,000 cleaned images and 137 labels.
2. [Original code and configurations](https://github.com/ML-Collective/Sign-to-Speech-for-Sign-Language-Understanding).
3. Optional: a small consented unseen-user test set for cross-signer/cross-condition evaluation.

## 7. Expected challenges

- Small dataset across many classes; possible class imbalance and similar-looking signs.
- Static frames omit movement, posture, and facial expression.
- Unseen signers, lighting, backgrounds, and camera angles may reduce accuracy.
- Signer metadata may be unavailable.
- Confidence thresholds may reduce incorrect speech but increase abstentions.
- A larger model may be accurate but too slow for real-time use.

## 8. Preliminary success metrics

| Area | Metrics |
|---|---|
| Recognition | Precision, recall, mAP50, mAP50-95, macro F1, per-class recall |
| Efficiency | FPS, inference latency, end-to-end latency, model size |
| Generalisation | Results on signer-disjoint or unseen-user/cross-condition test data |
| Reliability | Incorrect spoken outputs per 100 signing events; correct/incorrect abstentions; raw versus filtered output |

The paper's YOLOv5m results will be a historical reference; direct comparison requires the same split and training conditions.

## 9. Timeline and deployment resources

| Week | Activity | Resources |
|---|---|---|
| 1 | Audit dataset, labels, licence, and signer metadata | Dataset, annotation viewer |
| 2 | Prepare splits and train YOLO26n | Laptop/Colab GPU, Ultralytics |
| 3 | Train YOLO26s; compare accuracy and speed | GPU, experiment log |
| 4 | Build webcam, confidence, stability, and TTS pipeline | Webcam, speakers, OpenCV, `pyttsx3` |
| 5 | Run reliability and optional unseen-user tests | Test protocol, consent form if needed |
| 6 | Finalise report, architecture diagram, and demo | Report and presentation tools |

## References

Kolawole, S. et al. (2021). *Sign-to-Speech Model for Sign Language Understanding: A Case Study of Nigerian Sign Language.* [arXiv:2111.00995](https://arxiv.org/abs/2111.00995).  
[Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/).  
[United Nations SDG 10](https://sdgs.un.org/goals/goal10).
