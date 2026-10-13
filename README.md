# How to run the YOLO26 NSL prototype

## Required repository structure

This repository follows the Phase 1/Phase 2 assignment layout while adapting the
tabular examples to image object detection. Raw and processed data are deliberately
ignored by Git; download the authorised dataset locally into `data/raw/`.

```text
nsl_yolo26_project/
├── .gitignore
├── requirements.txt
├── README.md
├── data/{raw,processed}/
├── phase1_strategy/Phase1_Strategy_Report.md
├── phase2_implementation/
│   └── notebooks/pipeline_execution.ipynb
├── src/{__init__.py,transformers.py,pipeline_builder.py}
├── figures/
├── train.py
└── realtime_sign_to_speech.py
```

The final course submission will add `Phase1_Strategy_Report.pdf`,
`Phase2_Final_Report.pdf`, and eight generated 300-DPI diagnostic figures after
their underlying analysis has been executed. The Phase 1 report is intentionally
saved as Markdown now so it can be reviewed and revised before PDF export.

## What your professor meant by “try a new YOLO”

The supplied paper used **YOLOv5m**. Your professor is asking you to keep the same research problem but replace the detector with a newer YOLO generation and evaluate whether it is better. **YOLO26** is the current Ultralytics model family (released in 2026). It has a newer end-to-end detection design, a lighter detection head, and an updated training recipe. That does **not** mean it will automatically be better on this small NSL dataset: you must train and compare it fairly.

Use `yolo26n.pt` first for a quick, fast baseline, then `yolo26s.pt` for the main comparison. Do not compare a small YOLO26 model against the paper's larger YOLOv5m and claim a version-only improvement: report model size, training split, input size, hardware, FPS, and all metrics.

## Prepare the data

Download only where the dataset licence permits. Arrange data in YOLO detection format:

```text
data/nsl_yolo_dataset/
  images/train/    labels/train/
  images/val/      labels/val/
  images/test/     labels/test/
```

Every image needs a matching `.txt` file. Each line is `class_id x_center y_center width height`, with the final four values normalised to 0–1. Update `data/nsl.yaml` with absolute or correct relative paths and the full 137-class label list. Split by signer, not randomly by image.

## Install and train

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --data data/nsl.yaml --model yolo26n.pt --name yolo26n_baseline
python train.py --data data/nsl.yaml --model yolo26s.pt --name yolo26s_baseline
```

`train.py` loads a pretrained YOLO26 checkpoint, fine-tunes it on the labelled NSL images, saves learning curves and weights, then validates `best.pt`. The `--data` argument identifies the images and labels; `--imgsz 640` matches the paper's resized-image resolution; `--batch` controls memory use; `patience=20` avoids wasting training when validation stops improving.

## Run the demo

```bash
python realtime_sign_to_speech.py \
  --weights runs/nsl_yolo26/yolo26s_baseline/weights/best.pt
```

For each webcam frame, the script asks YOLO26 for detected boxes and class scores, draws the prediction, and keeps only the top-confidence label. It requires the same label in eight consecutive frames before speech. This temporal stabilisation is crucial: without it, a changing prediction would trigger repeated or incorrect speech. The confidence threshold and repeat cooldown can be adjusted, but set them using validation and user testing—not guesses.

## What to report

Compare YOLO26n and YOLO26s with the paper's reported YOLOv5m results (precision 0.806, recall 0.951, mAP50 0.953, mAP50-95 about 0.644). Use your own held-out split and label the paper figures as an external baseline, because different data splits make direct numeric comparison imperfect.

Report: mAP50-95, mAP50, precision, recall, macro F1, per-class errors, FPS, latency, model file size, device, and outcomes for different signers/backgrounds. Include examples of failures. A system-design project earns credibility by defining what it cannot do as well as what it can.

## System design

```text
Webcam frame → YOLO26 sign detector → confidence + stability filter → screen text → text-to-speech
                      │                         │
                      └── bounding box + label ─┘
```

This pipeline is intentionally simple for an initial demonstration. It is a sign-label recogniser followed by speech synthesis, not a complete language translator. The next research phase would collect consented video clips and use hand/body/face landmarks plus a temporal model for dynamic signs and sentences.
