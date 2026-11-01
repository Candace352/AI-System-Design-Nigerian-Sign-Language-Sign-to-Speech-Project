# Phase 1: Preprocessing and Feature-Engineering Strategy
## Nigerian Sign Language Sign-to-Speech Using YOLO26

### Scope note/adjustment: adapting the rubric to computer vision

This is a multi-class **object-detection** project, not a tabular prediction problem. The model consumes image pixels and predicts a sign class and bounding box. Therefore, tabular-only methods—such as Pearson screening of raw pixels, VIF, ordinal encoding of class labels, mean/median imputation, and `StandardScaler`—are not applied to the image tensor. Equivalent computer-vision controls are specified below: annotation validation, image-quality checks, class-balance analysis, duplicate detection, and leakage-safe dataset partitioning. The Pascal VOC values below were computed directly from the supplied archive on 14 October 2026; the separately packaged YOLO copy will not be merged with it because that would duplicate observations.

## A. Dataset and problem audit

### A1. Dataset dimensions and unit of analysis

The source dataset is the [Lanfrica/Hugging Face Nigerian Sign Language dataset](https://huggingface.co/datasets/Lanfrica/sign-to-speech-for-sign-language-understanding-a-case-study-of-nigerian-sign-language), associated with Kolawole et al. (2021). Its public description reports approximately 5,000 cleaned images covering 137 signs, words/phrases, and alphabet labels.

The primary observation is one annotated RGB image containing an isolated sign. The modelling tensor for each observation is an image resized with aspect-ratio-preserving letterboxing to **640 × 640 × 3**. The raw image collection is not naturally a two-dimensional table, so an audit manifest will be constructed with one row per annotated image/object.

The supplied `reference-original.zip` was inspected as a baseline implementation. It contains the original YOLOv5 model configuration, training logs, visual diagnostics, webcam client, and a `best.pt` model file, but **does not contain the source image/label dataset itself**. The subsequently supplied dataset archive does contain the raw images and annotations in two parallel representations: Pascal VOC (`train/` and `test/`, JPEG/XML) and YOLO (`dataset_YOLO/`, JPEG/TXT). The audit below uses the Pascal VOC representation as the raw source; the project will convert or validate one representation only before training, never concatenate both.

| Audit quantity | Definition | Phase 1 value |
|---|---|---|
| `N_images` | Readable Pascal VOC image files | 4,929 |
| `N_annotations` | Matched Pascal VOC XML annotation files | 4,929 |
| `N_objects` | Valid annotated sign boxes across all XML files | 5,204 |
| `P_manifest` | Core columns in the derived audit manifest | 11 |
| Class count | Unique labels observed and in the archive label map | 136 |
| Raw image dimensions | Image dimensions before letterboxing | 640 × 480 × 3 for all 4,929 images |
| Model input tensor | Pixels supplied to YOLO26 | 640 × 640 × 3 after letterbox preprocessing |

The 11 manifest columns are: `image_id`, `relative_path`, `image_width_px`, `image_height_px`, `channels`, `class_name`, `class_id`, `x_center`, `y_center`, `box_width`, and `box_height`. If the release includes signer, collection session, background, or lighting fields, they will be retained as audit metadata, not used as model inputs.

### A2. Variable inventory and data types

| Variable | Type | Role and treatment |
|---|---|---|
| Image pixels | Continuous bounded numeric, 0–255 | Main model input; converted by the YOLO pipeline to floating-point values in [0, 1] |
| Image width, height, channels | Discrete numeric | Audit only; expected positive dimensions and three channels |
| `x_center`, `y_center`, `box_width`, `box_height` | Continuous bounded numeric, 0–1 in YOLO format | Ground-truth detection target; validate range and geometry |
| `class_name` | Nominal categorical text | Human-readable target label |
| `class_id` | Nominal categorical integer | YOLO target identifier; integer values do **not** carry rank or magnitude |
| File path / image ID | Text / identifier | Traceability and duplicate/leakage checks; never a model feature |
| Signer/session/background/lighting, if supplied | Nominal categorical | Audit and group-splitting variables; never a main model feature |
| Timestamp, if supplied | Datetime | Audit only; no temporal split is planned unless it reveals data-collection leakage |

There are no ordinal predictors in the primary detector. In particular, NSL class IDs must **not** be ordinal encoded: `class_id=20` is merely a label identifier and is not greater than `class_id=4` in any mathematical sense.

### A3. Baseline missingness and annotation-integrity audit

Before cleaning, the audit script reported the following for the Pascal VOC source:

| Integrity check | Result | Rate |
|---|---:|---:|
| Readable RGB images | 4,929 / 4,929 | 100% |
| Image/XML pairs | 4,929 / 4,929 | 100% complete pairs |
| Images without XML annotation | 0 | 0% |
| XML annotations without image | 0 | 0% |
| Malformed XML files | 0 | 0% |
| Missing required filename, size, class, or box fields | 0 | 0% |
| Geometrically invalid boxes | 0 / 5,204 objects | 0% |
| Observed image mode and dimensions | RGB, 640 × 480 | 100% |

The supplied archive uses 136 label-map classes, despite the paper/repository description referring to 137 labels. This report uses the auditable archive value, **136**, and treats the documentation discrepancy as a data-version issue that must be stated in the final report.

The overall required-field missingness rate is therefore 0% for this Pascal VOC audit. Signer, session, background, and lighting metadata are not present in the inspected annotation schema; their availability cannot be inferred from filenames and is reported as unavailable rather than zero missing.

### A4. Target specification and class balance

The task is **multi-class object detection**, comprising (1) nominal classification of the NSL sign and (2) bounding-box localisation. It is not regression or ordinary image classification. The target is each valid pair of `class_id` and bounding box.

The Pascal VOC audit found 136 observed classes across 5,204 labelled objects. Per-class object counts range from **4** (`love`) to **72** (`D` and `K`), with a median of **37** and an imbalance ratio of **18:1**. The initial strategy is to preserve all valid minority-class images, stratify splits by class where feasible, and use conservative training-only augmentation. No synthetic labels will be created. If a class has too few examples for all three splits, that class will be reported as an evaluation limitation and a targeted consented data collection will be preferred over duplicating it into the test set.

## B. Cleaning, imputation, outliers, and association strategy

### B1. Missingness mechanisms and treatment

Missingness will be diagnosed from the audit rather than assumed:

| Situation | Provisional mechanism | Decision |
|---|---|---|
| Random file-transfer corruption affecting scattered files | Potential MCAR | Exclude unreadable files; log IDs and counts |
| Annotation/image failures concentrated in a collection session, signer, device, or background | Potential MAR | Investigate the observed grouping; report the affected subgroup and do not impute labels |
| Labels missing because difficult, ambiguous, or privacy-sensitive examples were omitted | Potential MNAR / unknown | Do not infer labels; quarantine and document potential selection bias |
| Missing optional signer or environment metadata | Unknown; may be MAR | Keep as `unknown` for audit reporting; do not fabricate group membership |

Images and labels are not imputed. A missing image cannot be meaningfully replaced by a mean image, and a missing sign label cannot be inferred without introducing label noise. A required image/annotation pair that fails validation will be excluded from training, validation, and testing with a reproducible exclusion log. Optional metadata is retained as missing/unknown and is not used as an input feature.

### B2. Image and annotation outlier protocol

Outlier checks will flag records for review, not automatically delete valid examples solely because they are unusual. The following deterministic rules will be applied:

| Check | Rule | Treatment |
|---|---|---|
| File validity | Image cannot be decoded, has zero width/height, or is not RGB after conversion | Exclude and log; no usable input exists |
| Bounding-box validity | `0 < width,height ≤ 1`; centre coordinates are in [0,1]; all box edges lie within image bounds | Repair only if a documented format conversion caused the error; otherwise exclude and log |
| Box geometry | Compute normalised box area and aspect ratio; flag values outside `Q1 − 1.5×IQR` to `Q3 + 1.5×IQR` | Visually inspect; retain plausible small/large signs because they represent real deployment conditions |
| Image exposure/blur | Flag luminance and blur scores outside 1.5×IQR; blur is measured by variance of the Laplacian | Visually inspect; retain realistic low-light or slightly blurred images unless the sign is not interpretable |
| Exact duplicates | SHA-256 hash match | Keep only one identical copy if labels agree; quarantine conflicting duplicates |
| Near duplicates | Perceptual hash distance threshold selected after audit | Place all members of a duplicate cluster in the same split; do not allow them to leak across splits |

The tabular Z-score rule (`|z| > 3`) is not the primary rule because box sizes, aspect ratios, brightness, and blur measures are expected to be skewed. IQR is more robust for these audit measures. No winsorisation/capping will be applied to image pixels or bounding-box labels: clipping would alter the visual evidence or move the true target. Valid extreme observations are retained; only objectively invalid files/annotations are removed.

**Observed duplicate leakage.** Exact SHA-256 matching found 114 duplicate groups containing 228 images (4.63% of the 4,929 Pascal VOC images). Every duplicate group crosses the archive's supplied train/test boundary. Consequently, the archive's original `train/` and `test/` folders are not valid for final evaluation and will not be used unchanged. Duplicate clusters must be treated as indivisible groups when creating new partitions.

### B3. Statistical association and multicollinearity protocol

Raw image pixels are high-dimensional spatial inputs, so pairwise Pearson correlation and VIF are not meaningful feature-selection tools for YOLO26. The detector learns visual features directly from the image. The following audit associations will instead be used to identify dataset bias and potential leakage:

| Audit relationship | Variable types | Diagnostic |
|---|---|---|
| Class label vs. box area/aspect ratio | Nominal vs. continuous | Kruskal–Wallis test plus descriptive distributions; used to identify class-specific framing bias |
| Class label vs. signer/background/session, if available | Nominal vs. nominal | Chi-square test and Cramér’s V; used to identify confounding/coverage gaps |
| Class label vs. brightness/blur category | Nominal vs. categorical/continuous | Chi-square or Kruskal–Wallis; used to identify acquisition-condition bias |
| Exact/near-duplicate cluster vs. dataset split | Identifier vs. split | Must be zero cross-split clusters; leakage check rather than a significance test |

These diagnostics are descriptive quality-control tests, not a basis for dropping visual information from the detector. If a later auxiliary **tabular** model is built from engineered metadata, only then will the following classical protocol apply: screen numeric features with `|r| ≥ 0.85`; calculate VIF after one-hot encoding; flag `VIF > 10`; and drop the feature with lower domain relevance, higher missingness, weaker target association, or redundant derivation. This protocol is not applied to raw image pixels or YOLO targets.

## C. Encoding, scaling, transformation, and partition strategy

### C1. Encoding and scaling

**Label encoding.** The class map will convert each NSL sign name to one integer `class_id` required by YOLO annotation files. This is nominal label indexing, not ordinal encoding. The map will be stored in version-controlled YAML and used identically in every split.

**Bounding-box transformation.** Bounding boxes will use YOLO's normalised format: `class_id x_center y_center width height`, where coordinates and dimensions are in [0,1]. Conversion from any source format will be unit-tested using image-boundary checks.

**Pixel scaling.** Images are read as 0–255 RGB values, letterboxed to 640×640 while preserving aspect ratio, and scaled to [0,1] by the YOLO26 preprocessing pipeline. `StandardScaler` is not appropriate because pixels are not independent Gaussian tabular variables, and per-dataset standardisation could conflict with the pretrained YOLO26 input pipeline. No additional MinMaxScaler will be fitted from validation or test data.

**Training-only augmentation.** After splitting, safe augmentations may include modest brightness/contrast/HSV variation, scale, and translation to improve robustness. Horizontal flipping will be disabled unless an NSL expert verifies that it cannot change the meaning of any included sign. Validation and test data receive only deterministic resize/letterbox preprocessing; they are never augmented.

### C2. Split and leakage-prevention design

The planned split is **70% training, 15% validation, and 15% test**, constructed anew after removing or grouping exact duplicates. The raw Pascal VOC source contains 4,929 images, so an ungrouped nominal allocation would be approximately 3,450/739/740 images; final counts may vary because duplicate clusters and class-stratification constraints take precedence over exact ratios.

1. **Stratification:** Maintain sign-class representation across train, validation, and test as far as class frequencies permit. The audit will identify any rare classes that cannot be represented in every partition.
2. **Group separation:** If signer IDs are available, all images from one signer will be assigned to one partition only. Group separation takes precedence over exact class proportions. If signer IDs are absent, duplicate/near-duplicate clusters and identifiable capture-session folders will be kept intact in one partition. A small external unseen-user test may be added later, with consent, as a robustness evaluation.
3. **Temporal alignment:** No chronological split is planned because the intended task is not time-series forecasting. Timestamps, if present, will only be inspected to detect collection-session leakage.
4. **Validation scheme:** YOLO26n and YOLO26s hyperparameters will be selected using the validation set only. The held-out test set remains sealed until the final evaluation. Each candidate configuration will use the same fixed split and seed; the final report will state the seed, split manifest, software version, hardware, and all selected settings. Full K-fold retraining is not planned because training multiple object-detection models is computationally expensive; the fixed validation partition is the internal selection loop.
5. **Fit isolation:** Dataset-derived choices—duplicate removal, augmentation settings, class-frequency decisions, confidence threshold, and temporal-stability parameters—will be selected using training/validation data only. The final test set will not influence preprocessing, model selection, thresholds, or error-analysis decisions.

## D. Audit of the supplied original implementation and proposed improvements

### D1. Evidence from `reference-original.zip`

The archive makes the original baseline sufficiently concrete to reproduce its *design intent* without treating its results as a directly comparable experiment:

| Baseline element | Evidence in archive | Implication for this project |
|---|---|---|
| Detector | `savedmodels_configs/yolo_model/opt.yaml` specifies `model: yolov5m` and `weights: yolov5m.pt` | YOLO26n and YOLO26s replace YOLOv5m as modern detector candidates |
| Input | `img_size: [640, 640]` | Retain 640×640 as the initial input size so that the new detector is assessed on a comparable image scale |
| Training budget | `epochs: 150`, `batch_size: 16`, eight workers | Start with a documented YOLO26 budget; report any necessary hardware-driven differences rather than hiding them |
| Class space | `nc: 137` and an explicit 137-name label list in `opt.yaml` | Preserve the original class-map ordering after confirming it matches the downloaded annotations |
| Original partitions | `train: dataset_YOLO/train`; `val: dataset_YOLO/test` | The original configuration has no separate validation partition. The proposed 70/15/15 split separates model selection from final testing |
| Reported final epoch | Epoch 149/149 log: precision 0.8057, recall 0.9512, mAP50 0.9533, mAP50-95 0.6439 | Use only as a historical reference; a direct comparison requires the same raw split, preprocessing, hardware, and evaluation code |
| Original augmentation | Hue/saturation/value adjustment, translation 0.1, scale 0.5, mosaic 1.0, and horizontal flip probability 0.5 | Re-evaluate all augmentation for NSL semantics; specifically, disable horizontal flipping unless an NSL expert confirms it preserves every label’s meaning |
| Live deployment | `livefeed_detection.py` resizes webcam images to width 400, submits every frame to a local DeepStack endpoint, and uses `pyttsx3` | The new system will run local YOLO26 inference and report end-to-end latency; the 400-pixel network-client path is not carried forward blindly |
| Speech policy | The original client speaks whenever a non-empty top detection is returned | The confidence-plus-temporal-stability layer is a defined system contribution: it reduces repetitive and low-confidence speech |

### D2. Baseline versus proposed system architecture

```text
Original reference implementation
webcam frame → resize to 400px width → DeepStack/YOLOv5m → top detection → speak every detected label

Proposed implementation
webcam frame → deterministic 640px letterbox → local YOLO26n or YOLO26s
             → confidence gate → temporal stability window → [speak label | abstain/request repeat]
```

The proposed design separates **detector quality** from **system-output reliability**. Detector metrics (precision, recall, mAP50, and mAP50-95) measure whether YOLO26 recognises and localises signs. Event-level metrics measure the user-facing policy: correct spoken output, incorrect spoken output, correct abstention, incorrect abstention, and end-to-end response time. This distinction prevents a high frame-level score from being mistaken for dependable speech behaviour.

### D3. Reliability-layer protocol

The live inference system will use a confidence threshold `τ` and a fixed temporal window of `W` frames. For each frame, it retains the highest-confidence detection, provided its confidence is at least `τ`.

1. Keep the most recent `W` accepted labels and confidences.
2. Speak a label only when the same label is present in at least `K` of the `W` frames and its aggregated confidence exceeds `τ`.
3. Do not repeat a just-spoken label until a cooldown period has elapsed or a different stable label appears.
4. If there is no qualifying prediction, display an abstention message such as “Please repeat the sign”; do not generate an unsupported spoken label.

Initial values (`τ = 0.65`, `W = 8`, `K = 8`, and a two-second cooldown) are starting points, not final claims. They will be selected using validation events only. The final test set will evaluate the locked policy once. The analysis will report the reliability–latency trade-off: stricter settings may reduce incorrect speech but increase abstentions or response delay.

### D4. Figures and evidence planned for Phase 1

The Phase 1 submission will save the following reproducible diagnostics under `figures/` after audit:

1. Missing/invalid-image and annotation summary table.
2. Per-class frequency bar chart and imbalance ratio.
3. Bounding-box area and aspect-ratio distributions with IQR flags.
4. Image brightness and blur distributions with reviewed flags.
5. Exact/near-duplicate audit summary and split-cluster leakage check.
6. Train/validation/test class-distribution comparison.
7. Sample annotation montage, including representative valid, flagged, and excluded records.

The original archive's `labels.jpg`, `labels_correlogram.jpg`, training batches, test batches, confusion matrix, precision–recall curve, and training-results plot will be retained as **reference baseline artifacts**, not re-labelled as results from this project.

### Reproducibility and deliverables

The repository will retain the raw dataset outside Git tracking, a versioned class map, split-manifest CSV files, an exclusion log, audit tables, and saved figures. `data/raw/`, `data/processed/`, model weights, and cache files will be ignored by Git. The README will document the dataset source, licence, setup, preprocessing decisions, and exact commands needed to reproduce the split and training workflow.

## References

- Kolawole, S., Osakuade, O., Saxena, N., & Olorisade, B. K. (2021). *Sign-to-Speech Model for Sign Language Understanding: A Case Study of Nigerian Sign Language.* [arXiv:2111.00995](https://arxiv.org/abs/2111.00995).
- [Lanfrica / Hugging Face dataset](https://huggingface.co/datasets/Lanfrica/sign-to-speech-for-sign-language-understanding-a-case-study-of-nigerian-sign-language).
- [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/).
