# Grad-CAM setup for RetinaCare AI

The uploaded project did not contain `gradcam_features.tflite` or `gradcam_dense_weights.bin`.
The updated Android code now uses a single feature-extractor inference and a smooth colour overlay,
but those two support assets must be generated from the included original Keras model first.

## Generate the two support assets

1. Open Google Colab (or the same Python environment used for the original Keras model).
2. Upload/extract this project ZIP so that `model_source/aptos_binary_efficientnetb0.keras` is available.
3. Run:

   ```bash
   pip install tensorflow numpy
   python tools/export_gradcam_assets.py
   ```

4. The script creates:
   - `gradcam_export/gradcam_features.tflite`
   - `gradcam_export/gradcam_dense_weights.bin`
5. Check that it prints feature output `(1, 7, 7, 1280)` and Dense weights `(1280, 2)`.
6. Copy both files into `app/src/main/assets/` beside the two existing `.tflite` files.
7. Reopen Android Studio, Sync Gradle, then Build > Assemble Project and Run.

The script uses the supplied model's `top_activation` layer and `predictions` Dense kernel. In this
model, the classifier head is GlobalAveragePooling -> Dropout -> Dense. For the class logit, the
Dense kernel supplies the channel weights used for the class activation map. The model's original
preprocessing graph is retained in the feature extractor.

## Stability changes

The app now performs image validation, blur screening, DR prediction and Grad-CAM on a single
background executor. The image picker is disabled during processing. The previous 49-pass
occlusion routine and its repeated temporary bitmap allocations have been removed.

## Important

This is a research prototype, not a clinical diagnostic device. The heatmap is an explanatory
visualization and must not be interpreted as a lesion segmentation or proof of disease.
