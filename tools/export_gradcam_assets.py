"""Export Grad-CAM support assets from the existing trained EfficientNet model.

Run in the same TensorFlow/Keras environment used to save the .keras model.
Outputs:
  gradcam_features.tflite      feature tensor [1, 7, 7, 1280]
  gradcam_dense_weights.bin    Dense kernel [1280, 2], little-endian float32
"""
from pathlib import Path
import numpy as np
import tensorflow as tf

MODEL_PATH = Path("model_source/aptos_binary_efficientnetb0.keras")
OUT_DIR = Path("gradcam_export")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Put the original Keras model at this path."
    )

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# These names were verified in the supplied model architecture.
feature_layer = model.get_layer("top_activation")
classifier = model.get_layer("predictions")

feature_model = tf.keras.Model(
    inputs=model.input,
    outputs=feature_layer.output,
    name="efficientnet_gradcam_features",
)

kernel, bias = classifier.get_weights()
if kernel.shape != (1280, 2):
    raise ValueError(f"Expected classifier kernel (1280, 2), got {kernel.shape}")

# The model's own input preprocessing layers remain inside feature_model.
converter = tf.lite.TFLiteConverter.from_keras_model(feature_model)
converter.optimizations = []  # Keep float32 tensors and predictable output shape.
tflite_bytes = converter.convert()
(OUT_DIR / "gradcam_features.tflite").write_bytes(tflite_bytes)

# Dense kernel is stored row-major: [channel, class], little-endian float32.
np.asarray(kernel, dtype="<f4").tofile(OUT_DIR / "gradcam_dense_weights.bin")

# Validate exported feature tensor before copying to Android.
interpreter = tf.lite.Interpreter(model_path=str(OUT_DIR / "gradcam_features.tflite"))
interpreter.allocate_tensors()
in_detail = interpreter.get_input_details()[0]
out_detail = interpreter.get_output_details()[0]
print("Input:", in_detail["shape"], in_detail["dtype"])
print("Feature output:", out_detail["shape"], out_detail["dtype"])
print("Dense weights:", kernel.shape, "bytes:", (OUT_DIR / "gradcam_dense_weights.bin").stat().st_size)

assert tuple(in_detail["shape"]) == (1, 224, 224, 3)
assert tuple(out_detail["shape"]) == (1, 7, 7, 1280)
assert in_detail["dtype"] == np.float32 and out_detail["dtype"] == np.float32
print("Export completed. Copy both files from gradcam_export/ into app/src/main/assets/.")
