"""Computer-vision data validators used before YOLO26 training.

The assignment's suggested `transformers.py` is designed for tabular Scikit-Learn
pipelines. This project instead validates image/YOLO-label records. These utilities
are deliberately deterministic so each exclusion can be logged and reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AnnotationIssue:
    """A reproducible reason an annotation is unsuitable for training."""

    line_number: int
    message: str


def parse_yolo_line(line: str) -> tuple[int, float, float, float, float]:
    """Parse one `class x_center y_center width height` YOLO label line."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError("YOLO label must contain exactly five whitespace-separated values.")
    class_id = int(parts[0])
    x_center, y_center, width, height = (float(value) for value in parts[1:])
    return class_id, x_center, y_center, width, height


def validate_yolo_annotations(lines: Iterable[str], class_count: int) -> list[AnnotationIssue]:
    """Return every structural/range error without silently repairing labels."""
    issues: list[AnnotationIssue] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            class_id, x, y, width, height = parse_yolo_line(line)
        except (TypeError, ValueError) as error:
            issues.append(AnnotationIssue(line_number, str(error)))
            continue

        if not 0 <= class_id < class_count:
            issues.append(AnnotationIssue(line_number, f"class_id {class_id} is outside 0–{class_count - 1}."))
        if not (0 < width <= 1 and 0 < height <= 1):
            issues.append(AnnotationIssue(line_number, "width and height must be in (0, 1]."))
        if not (0 <= x <= 1 and 0 <= y <= 1):
            issues.append(AnnotationIssue(line_number, "centre coordinates must be in [0, 1]."))
        if x - width / 2 < 0 or x + width / 2 > 1 or y - height / 2 < 0 or y + height / 2 > 1:
            issues.append(AnnotationIssue(line_number, "bounding box extends beyond image boundaries."))
    return issues


def box_area(width: float, height: float) -> float:
    """Return normalised bounding-box area for IQR-based audit diagnostics."""
    return width * height
