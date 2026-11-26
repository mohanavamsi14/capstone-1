package com.example.droffline

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {
    private lateinit var interpreter: Interpreter
    private lateinit var retinaValidator: RetinaValidator
    private lateinit var gradCam: GradCam
    private lateinit var preview: ImageView
    private lateinit var explanation: ImageView
    private lateinit var result: TextView
    private lateinit var scores: TextView
    private lateinit var chooseButton: Button

    private val worker = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val processing = AtomicBoolean(false)
    private val imageSize = 224
    private val retinaThreshold = 0.5f

    private val picker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) analyze(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        preview = findViewById(R.id.previewImage)
        explanation = findViewById(R.id.explanationImage)
        result = findViewById(R.id.resultText)
        scores = findViewById(R.id.scoresText)
        chooseButton = findViewById(R.id.chooseButton)

        try {
            interpreter = Interpreter(loadModel("diabetic_retinopathy.tflite"), Interpreter.Options().setNumThreads(2))
            retinaValidator = RetinaValidator(this)
            gradCam = GradCam(this)
            result.text = "Models loaded. Choose a retinal image."
            scores.text = "Retinal validation, blur screening and Grad-CAM are ready."
        } catch (e: Exception) {
            result.text = "Model loading failed"
            scores.text = e.message ?: "Check the required model files in app/src/main/assets."
            chooseButton.isEnabled = false
        }

        chooseButton.setOnClickListener {
            if (processing.get()) return@setOnClickListener
            if (!::interpreter.isInitialized || !::retinaValidator.isInitialized || !::gradCam.isInitialized) {
                result.text = "Models are not ready."
                return@setOnClickListener
            }
            picker.launch("image/*")
        }
    }

    private fun loadModel(fileName: String): MappedByteBuffer {
        val fd = assets.openFd(fileName)
        return try {
            FileInputStream(fd.fileDescriptor).channel.use { channel ->
                channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        } finally {
            fd.close()
        }
    }

    private fun analyze(uri: Uri) {
        if (!processing.compareAndSet(false, true)) return
        chooseButton.isEnabled = false
        result.text = "Checking image..."
        scores.text = "Please wait. Image validation and analysis are running."
        explanation.setImageDrawable(null)

        worker.execute {
            var source: Bitmap? = null
            var resized: Bitmap? = null
            try {
                val loadedBitmap = contentResolver.openInputStream(uri)?.use { stream ->
                    BitmapFactory.decodeStream(stream)
                } ?: throw IllegalArgumentException("Cannot read selected image.")
                source = loadedBitmap

                mainHandler.post { preview.setImageBitmap(loadedBitmap) }

                val retinalScore = retinaValidator.retinalProbability(loadedBitmap)
                if (!retinalScore.isFinite() || retinalScore !in 0f..1f) {
                    throw IllegalStateException("Retinal validator returned an invalid score.")
                }
                if (retinalScore < retinaThreshold) {
                    showResult("Invalid Image!", String.format(Locale.US,
                        "Please upload a retinal fundus image.\nRetinal confidence: %.2f%%",
                        retinalScore * 100f))
                    return@execute
                }

                val blurScore = BlurDetector.calculateVariance(loadedBitmap)
                if (blurScore < BlurDetector.BLUR_THRESHOLD) {
                    showResult("Blur Detected!", String.format(Locale.US,
                        "Please upload a clearer retinal image.\nRetinal confidence: %.2f%%\nSharpness score: %.2f",
                        retinalScore * 100f, blurScore))
                    return@execute
                }

                val modelBitmap = Bitmap.createScaledBitmap(loadedBitmap, imageSize, imageSize, true)
                resized = modelBitmap
                val probabilities = predict(modelBitmap)
                require(probabilities.size >= 2) { "DR model must return two probabilities." }
                val dr = probabilities[0].coerceIn(0f, 1f)
                val noDr = probabilities[1].coerceIn(0f, 1f)
                val predictedClass = if (dr >= noDr) 0 else 1

                val explanationBitmap = gradCam.createOverlay(modelBitmap, predictedClass)
                val predictionText = if (predictedClass == 0) "Prediction: DR" else "Prediction: No DR"
                val details = String.format(Locale.US,
                    "Image validation: Passed\nRetinal confidence: %.2f%%\nSharpness score: %.2f\n\nDR score: %.2f%%\nNo DR score: %.2f%%\n\nResearch screening result only; not a diagnosis.",
                    retinalScore * 100f, blurScore, dr * 100f, noDr * 100f)

                mainHandler.post {
                    if (!isFinishing && !isDestroyed) {
                        result.text = predictionText
                        scores.text = details
                        explanation.setImageBitmap(explanationBitmap)
                    } else {
                        explanationBitmap.recycle()
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("RETINACARE", "Image analysis failed", e)
                showResult("Image analysis failed", e.localizedMessage ?: "Unexpected error. Please try another image.")
            } finally {
                resized?.recycle()
                // source remains referenced by the preview ImageView; do not recycle it here.
                mainHandler.post {
                    processing.set(false)
                    if (!isFinishing && !isDestroyed) chooseButton.isEnabled = true
                }
            }
        }
    }

    private fun showResult(title: String, detail: String) {
        mainHandler.post {
            if (!isFinishing && !isDestroyed) {
                result.text = title
                scores.text = detail
                explanation.setImageDrawable(null)
            }
        }
    }

    private fun predict(bitmap: Bitmap): FloatArray {
        val input = toInput(bitmap)
        val output = Array(1) { FloatArray(2) }
        interpreter.run(input, output)
        return output[0]
    }

    // Existing EfficientNet input contract: RGB float32 values in [0,255].
    private fun toInput(source: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(imageSize * imageSize * 3 * 4).order(ByteOrder.nativeOrder())
        for (y in 0 until imageSize) for (x in 0 until imageSize) {
            val px = source.getPixel(x, y)
            buffer.putFloat(((px shr 16) and 255).toFloat())
            buffer.putFloat(((px shr 8) and 255).toFloat())
            buffer.putFloat((px and 255).toFloat())
        }
        buffer.rewind()
        return buffer
    }

    override fun onDestroy() {
        // Queue interpreter cleanup after any in-flight analysis task to avoid closing
        // a TensorFlow Lite interpreter while it is being used on the worker thread.
        worker.execute {
            if (::interpreter.isInitialized) interpreter.close()
            if (::retinaValidator.isInitialized) retinaValidator.close()
            if (::gradCam.isInitialized) gradCam.close()
        }
        worker.shutdown()
        super.onDestroy()
    }
}
