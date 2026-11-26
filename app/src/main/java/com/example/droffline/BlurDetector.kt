
package com.example.droffline

import android.graphics.Bitmap
import kotlin.math.max

object BlurDetector {

    // Initial threshold only. Calibrate using your own clear and
    // blurred retinal images before relying on this value.
    const val BLUR_THRESHOLD = 40.0

    fun calculateVariance(source: Bitmap): Double {
        val maxDimension = 512

        val scale = maxDimension.toFloat() /
                max(source.width, source.height).toFloat()

        val width = (source.width * scale).toInt().coerceAtLeast(3)
        val height = (source.height * scale).toInt().coerceAtLeast(3)

        val bitmap = Bitmap.createScaledBitmap(
            source, width, height, true
        )

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val gray = IntArray(width * height)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (pixel shr 16) and 255
            val g = (pixel shr 8) and 255
            val b = pixel and 255

            gray[i] = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
        }

        var sum = 0.0
        var sumSquared = 0.0
        var count = 0

        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                val index = y * width + x

                val laplacian =
                    gray[index - width] +
                            gray[index + width] +
                            gray[index - 1] +
                            gray[index + 1] -
                            4 * gray[index]

                val value = laplacian.toDouble()

                sum += value
                sumSquared += value * value
                count++
            }
        }

        if (count == 0) return 0.0

        val mean = sum / count
        val variance = (sumSquared / count) - (mean * mean)
        if (bitmap !== source) bitmap.recycle()
        return variance
    }

    fun isBlurry(source: Bitmap): Boolean {
        return calculateVariance(source) < BLUR_THRESHOLD
    }
}
