from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
source = ROOT / "model_source" / "aptos_binary_efficientnetb0.keras"
target = ROOT / "app" / "src" / "main" / "assets" / "diabetic_retinopathy.tflite"
target.parent.mkdir(parents=True, exist_ok=True)

if not source.exists():
    raise FileNotFoundError(f"Model not found: {source}")

model = tf.keras.models.load_model(source, compile=False)
print("Model input:", model.input_shape)
print("Model output:", model.output_shape)

if tuple(model.input_shape[1:]) != (224, 224, 3):
    raise ValueError("Expected input shape (None, 224, 224, 3)")
if model.output_shape[-1] != 2:
    raise ValueError("Expected two outputs: DR and No_DR")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
target.write_bytes(tflite)
print(f"Created: {target} ({len(tflite)/1024/1024:.1f} MB)")
print("Important: validate TFLite predictions against the Keras model before research use.")
