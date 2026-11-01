# Diagnostic Figures

The `figures/` directory will contain 300-DPI PNG or SVG diagnostics generated from the full dataset-validation run. These figures evaluate image–annotation integrity and potential leakage; they are not model-performance results.

1. `fig1_1_missingness_heatmap.png` — image, annotation, and metadata completeness.
2. `fig1_2_target_distribution.png` — class frequencies across the 137 NSL labels.
3. `fig2_1_outlier_boxplots.png` — bounding-box area, aspect ratio, brightness, or blur outliers.
4. `fig3_1_annotation_association_matrix.png` — relationships among audit features, such as box geometry and image characteristics.
5. `fig3_2_duplicate_cluster_audit.png` — exact and candidate near-duplicate groups, used to check split leakage.
6. `fig3_3_class_condition_bias.png` — class-level differences in box geometry or image conditions.
7. `fig4_1_augmentation_grid.png` — original images compared with safe, training-only augmentation examples.
8. `fig5_1_box_area_aspect_ratio.png` — distribution of bounding-box area and aspect ratio.

The assignment’s tabular correlation, VIF, and PCA examples do not apply directly to raw YOLO image tensors. Therefore, they are replaced with computer-vision diagnostics that assess annotation quality, class imbalance, image conditions, and data leakage. If PCA is explicitly required by the instructor, it will be applied only to an appropriate derived representation, such as training-set audit features or pretrained image embeddings—not raw pixel columns.
