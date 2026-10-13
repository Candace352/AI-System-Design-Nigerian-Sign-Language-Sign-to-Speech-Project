"""Fine-tune YOLO26 for the isolated Nigerian Sign Language detection task.

Example:
    python train.py --data data/nsl.yaml --model yolo26s.pt --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO26 NSL detector.")
    parser.add_argument("--data", default="data/nsl.yaml", help="Dataset YAML file.")
    parser.add_argument("--model", default="yolo26s.pt", help="Pretrained YOLO26 checkpoint.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="Lower this if GPU memory is limited.")
    parser.add_argument("--device", default=None, help="For example: 0, mps, or cpu.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/nsl_yolo26")
    parser.add_argument("--name", default="yolo26s_baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.data).is_file():
        raise FileNotFoundError(
            f"Cannot find {args.data}. Create the YOLO dataset folders and update data/nsl.yaml first."
        )

    # The pretrained checkpoint supplies general visual features; training adapts it to NSL labels.
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=20,       # Stop if validation no longer improves for 20 epochs.
        pretrained=True,
        plots=True,
        seed=42,
    )

    # This evaluates the saved best checkpoint on the YAML's validation split.
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.is_file():
        metrics = YOLO(best_weights).val(data=args.data, imgsz=args.imgsz, split="val")
        print(f"mAP50-95: {metrics.box.map:.3f}; mAP50: {metrics.box.map50:.3f}")


if __name__ == "__main__":
    main()
