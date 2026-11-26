package com.example.droffline

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * Class Activation Map for the EfficientNetB0 GAP -> Dense classifier.
 * For this linear GAP head, the Dense kernel gives the channel weights
 * used by Grad-CAM for the class logit (before softmax).
 */
class GradCam(context: Context) {
    private val interpreter: Interpreter
    private val channels = 1280
    private val classes = 2
    private val featureSize = 7
    private val denseWeights: FloatArray

    init {
        val fd = context.assets.openFd("gradcam_features.tflite")
        val modelBuffer = FileInputStream(fd.fileDescriptor).channel.use { channel ->
            channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
        fd.close()
        interpreter = Interpreter(modelBuffer, Interpreter.Options().setNumThreads(2))

        val shape = interpreter.getOutputTensor(0).shape()
        require(shape.contentEquals(intArrayOf(1, featureSize, featureSize, channels))) {
            "Unexpected Grad-CAM feature output: ${shape.contentToString()}"
        }

        val weightBytes = context.assets.open("gradcam_dense_weights.bin").use { it.readBytes() }
        require(weightBytes.size == channels * classes * 4) {
            "Invalid Grad-CAM dense weights file: ${weightBytes.size} bytes"
        }
        val buffer = ByteBuffer.wrap(weightBytes).order(ByteOrder.LITTLE_ENDIAN)
        denseWeights = FloatArray(channels * classes) { buffer.float }
    }

    /** source must be 224x224 and use the DR model's original RGB [0,255] input. */
    fun createOverlay(source: Bitmap, classIndex: Int): Bitmap {
        require(classIndex in 0 until classes) { "Invalid class index: $classIndex" }
        val input = ByteBuffer.allocateDirect(224 * 224 * 3 * 4).order(ByteOrder.nativeOrder())
        for (y in 0 until 224) for (x in 0 until 224) {
            val pixel = source.getPixel(x, y)
            input.putFloat(((pixel shr 16) and 255).toFloat())
            input.putFloat(((pixel shr 8) and 255).toFloat())
            input.putFloat((pixel and 255).toFloat())
        }
        input.rewind()

        val features = Array(1) { Array(featureSize) { Array(featureSize) { FloatArray(channels) } } }
        interpreter.run(input, features)

        val cam = Array(featureSize) { FloatArray(featureSize) }
        var maxValue = 0f
        for (y in 0 until featureSize) {
            for (x in 0 until featureSize) {
                var sum = 0f
                for (c in 0 until channels) {
                    sum += features[0][y][x][c] * denseWeights[c * classes + classIndex]
                }
                val positive = max(0f, sum)
                cam[y][x] = positive
                maxValue = max(maxValue, positive)
            }
        }

        val heatPixels = IntArray(224 * 224)
        for (y in 0 until 224) {
            val gy = y.toFloat() * (featureSize - 1) / 223f
            val y0 = gy.toInt().coerceIn(0, featureSize - 1)
            val y1 = min(y0 + 1, featureSize - 1)
            val fy = gy - y0
            for (x in 0 until 224) {
                val gx = x.toFloat() * (featureSize - 1) / 223f
                val x0 = gx.toInt().coerceIn(0, featureSize - 1)
                val x1 = min(x0 + 1, featureSize - 1)
                val fx = gx - x0
                val a = cam[y0][x0] * (1f - fx) + cam[y0][x1] * fx
                val b = cam[y1][x0] * (1f - fx) + cam[y1][x1] * fx
                val value = if (maxValue > 0f) ((a * (1f - fy) + b * fy) / maxValue).coerceIn(0f, 1f) else 0f
                heatPixels[y * 224 + x] = jetColor(value)
            }
        }

        val heatBitmap = Bitmap.createBitmap(heatPixels, 224, 224, Bitmap.Config.ARGB_8888)
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply { alpha = 105 }
        canvas.drawBitmap(heatBitmap, Rect(0, 0, 224, 224), Rect(0, 0, 224, 224), paint)
        heatBitmap.recycle()
        return result
    }

    private fun jetColor(v: Float): Int {
        val r = (1.5f - kotlin.math.abs(4f * v - 3f)).coerceIn(0f, 1f)
        val g = (1.5f - kotlin.math.abs(4f * v - 2f)).coerceIn(0f, 1f)
        val b = (1.5f - kotlin.math.abs(4f * v - 1f)).coerceIn(0f, 1f)
        return android.graphics.Color.rgb((r * 255).toInt(), (g * 255).toInt(), (b * 255).toInt())
    }

    fun close() = interpreter.close()
}
