"""Audit a Pascal VOC image/annotation dataset stored in a ZIP archive.

Usage:
    python -m src.audit_pascal_voc_zip /path/to/dataset.zip

The script prints JSON so its results can be copied into the Phase 1 report. It
does not extract, alter, or upload the source data.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def path_key(path: str) -> tuple[str, str]:
    """Return the partition and filename stem for a train/test archive member."""
    parts = PurePosixPath(path).parts
    partition = next(part for part in parts if part in {"train", "test"})
    return partition, PurePosixPath(path).stem


def optional_text(node: ET.Element | None, tag: str) -> str | None:
    child = node.find(tag) if node is not None else None
    return child.text.strip() if child is not None and child.text else None


def audit(archive_path: str, hash_images: bool = True) -> dict:
    """Return integrity, annotation, and class-balance statistics for train/test VOC files."""
    images: dict[tuple[str, str], str] = {}
    annotations: dict[tuple[str, str], str] = {}
    image_errors: list[str] = []
    xml_errors: list[str] = []
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    invalid_boxes = 0
    object_count = 0
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    expected_classes: list[str] = []

    with zipfile.ZipFile(archive_path) as archive:
        members = [
            item.filename
            for item in archive.infolist()
            if not item.is_dir() and not item.filename.startswith("__MACOSX/")
        ]
        label_map_member = next((member for member in members if member.endswith("label_map.pbtxt")), None)
        if label_map_member:
            label_map_text = archive.read(label_map_member).decode("utf-8", errors="replace")
            expected_classes = re.findall(r"name\s*:\s*'([^']+)'", label_map_text)
        for member in members:
            suffix = PurePosixPath(member).suffix.lower()
            if suffix not in IMAGE_SUFFIXES | {".xml"}:
                continue
            # The release also includes a separate `dataset_YOLO/` copy. It must
            # not be mixed with Pascal VOC files or duplicate-format records will
            # inflate the audit counts and create apparent leakage.
            if "dataset_YOLO" in PurePosixPath(member).parts:
                continue
            try:
                partition, stem = path_key(member)
            except StopIteration:
                continue
            if suffix in IMAGE_SUFFIXES:
                images[(partition, stem)] = member
            else:
                annotations[(partition, stem)] = member

        for key, member in images.items():
            try:
                payload = archive.read(member)
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
                with Image.open(io.BytesIO(payload)) as image:
                    dimensions[f"{image.width}x{image.height}"] += 1
                    modes[image.mode] += 1
                if hash_images:
                    hashes[hashlib.sha256(payload).hexdigest()].append(member)
            except Exception as error:  # Report malformed images instead of hiding them.
                image_errors.append(f"{member}: {error}")

        for key, member in annotations.items():
            try:
                root = ET.fromstring(archive.read(member))
            except ET.ParseError as error:
                xml_errors.append(f"{member}: {error}")
                continue

            for field in ("filename",):
                if not optional_text(root, field):
                    missing_fields[field] += 1
            size = root.find("size")
            width = optional_text(size, "width")
            height = optional_text(size, "height")
            if not width:
                missing_fields["width"] += 1
            if not height:
                missing_fields["height"] += 1

            for obj in root.findall("object"):
                object_count += 1
                name = optional_text(obj, "name")
                if not name:
                    missing_fields["class_name"] += 1
                else:
                    classes[name] += 1
                box = obj.find("bndbox")
                coordinates = [optional_text(box, field) for field in ("xmin", "ymin", "xmax", "ymax")]
                if any(value is None for value in coordinates):
                    missing_fields["bounding_box_coordinate"] += 1
                    invalid_boxes += 1
                    continue
                try:
                    xmin, ymin, xmax, ymax = (float(value) for value in coordinates)
                    image_width, image_height = float(width), float(height)
                    if not (0 <= xmin < xmax <= image_width and 0 <= ymin < ymax <= image_height):
                        invalid_boxes += 1
                except (TypeError, ValueError):
                    invalid_boxes += 1

    paired = set(images) & set(annotations)
    exact_duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    cross_partition_duplicate_groups = sum(
        len({path_key(member)[0] for member in group}) > 1
        for group in exact_duplicate_groups
    )
    by_partition = {
        partition: {
            "images": sum(key[0] == partition for key in images),
            "annotations": sum(key[0] == partition for key in annotations),
            "complete_pairs": sum(key[0] == partition for key in paired),
        }
        for partition in ("train", "test")
    }
    return {
        "archive": archive_path,
        "partitions": by_partition,
        "totals": {
            "images": len(images),
            "annotations": len(annotations),
            "complete_pairs": len(paired),
            "images_without_annotation": len(set(images) - set(annotations)),
            "annotations_without_image": len(set(annotations) - set(images)),
            "objects": object_count,
            "classes_observed": len(classes),
            "unreadable_images": len(image_errors),
            "malformed_xml": len(xml_errors),
            "invalid_boxes": invalid_boxes,
            "exact_duplicate_groups": len(exact_duplicate_groups),
            "images_in_exact_duplicate_groups": sum(len(group) for group in exact_duplicate_groups),
            "exact_duplicate_groups_cross_partition": cross_partition_duplicate_groups,
            "label_map_classes": len(expected_classes),
        },
        "image_dimensions": dict(dimensions),
        "image_modes": dict(modes),
        "class_counts": dict(classes.most_common()),
        "label_map_classes_without_pascal_voc_annotation": sorted(set(expected_classes) - set(classes)),
        "missing_annotation_fields": dict(missing_fields),
        "examples": {
            "image_errors": image_errors[:10],
            "xml_errors": xml_errors[:10],
            "exact_duplicate_groups": exact_duplicate_groups[:10],
        },
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m src.audit_pascal_voc_zip /path/to/dataset.zip")
    print(json.dumps(audit(sys.argv[1]), indent=2, sort_keys=True))
