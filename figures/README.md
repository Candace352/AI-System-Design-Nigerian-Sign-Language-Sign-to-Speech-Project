# Diagnostic figures

Phase 2 will generate and save these files after the raw-data audit:

1. `fig1_1_missingness_heatmap.png`
2. `fig1_2_target_distribution.png`
3. `fig2_1_outlier_boxplots.png`
4. `fig3_1_annotation_association_matrix.png` *(adapted from a tabular correlation matrix)*
5. `fig3_2_duplicate_cluster_audit.png` *(adapted from collinearity scatterplots)*
6. `fig3_3_class_condition_bias.png` *(adapted from target-importance figure)*
7. `fig4_1_augmentation_grid.png` *(adapted from transformation grid)*
8. `fig5_1_box_area_aspect_ratio.png` *(PCA scree plot is not appropriate for raw YOLO image inputs)*

All final PNG/SVG files will be exported at 300 DPI. The original tabular correlation, VIF, and PCA figures are replaced by image-dataset diagnostics because the model is a YOLO object detector, not a tabular feature pipeline.
