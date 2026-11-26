# Diabetic Retinopathy Android App (offline)

This Android Studio project uses the supplied trained EfficientNetB0 binary model.
It predicts DR / No DR and displays an on-device **occlusion-sensitivity explanation heatmap**.

## Important explainability note
This build does NOT claim to implement gradient-based Grad-CAM. Standard TensorFlow Lite
inference does not expose TensorFlow/Keras GradientTape gradients. The included red overlay
is occlusion sensitivity: image patches are masked and the change in the predicted class score
is measured. It is an explainability approximation, not Grad-CAM and not a medical rationale.

## 1. Convert the included model
Install Python 3.10 or 3.11 in a virtual environment, then:
```bash
python -m pip install tensorflow==2.15.1 numpy==1.26.4
python convert_model.py
```
The script writes `app/src/main/assets/diabetic_retinopathy.tflite`.

## 2. Build in Android Studio
- Open this folder in Android Studio.
- Allow Gradle sync (internet may be required the first time to download dependencies).
- Connect an Android device or start an emulator.
- Press Run, or use Build > Build Bundle(s) / APK(s) > Build APK(s).

## Model assumptions
- Input: float32 `[1,224,224,3]`, RGB values 0..255.
- Output: two scores, index 0 = DR, index 1 = No_DR.
- The Keras EfficientNet model is expected to include its own input rescaling.
Validate TFLite output against the original Keras model before relying on results.

## Offline use
After model conversion, Gradle dependencies are downloaded, and the app is installed,
image prediction and occlusion heatmap generation run locally without internet.

## Limitations
- Binary DR / No DR only; not grades 0–4.
- The included model was trained for a research project and is not clinically validated.
- No camera capture or prediction history in this starter.
- Heatmap generation runs 49 additional inferences per image and may take several seconds.

## Updated stability and Grad-CAM implementation

The updated `MainActivity.kt` moves image validation, DR inference and heatmap generation off the UI thread. It disables the image picker while a job is running and removes the previous 49-pass occlusion-map routine. The explanation view now uses a smooth class activation overlay rather than red grid boxes.

Before running this updated project, follow `GRADCAM_SETUP.md` to generate and copy these two required assets from the included Keras model:

- `app/src/main/assets/gradcam_features.tflite`
- `app/src/main/assets/gradcam_dense_weights.bin`

The two existing model assets (`diabetic_retinopathy.tflite` and `retina_validator.tflite`) are retained.
"# capstone-1" 
