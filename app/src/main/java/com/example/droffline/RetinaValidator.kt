
package com.example.droffline

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class RetinaValidator(context: Context) {

    private val interpreter: Interpreter
    private val imageSize = 224

    init {
        val fd = context.assets.openFd("retina_validator.tflite")

        val modelBuffer = FileInputStream(fd.fileDescriptor).channel.use {
            it.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength
            )
        }

        fd.close()

        interpreter = Interpreter(
            modelBuffer,
            Interpreter.Options().setNumThreads(4)
        )

        require(
            interpreter.getInputTensor(0).shape().contentEquals(
                intArrayOf(1, 224, 224, 3)
            )
        ) {
            "Incorrect validator input shape"
        }
    }

    fun retinalProbability(source: Bitmap): Float {

        val bitmap = Bitmap.createScaledBitmap(
            source,
            imageSize,
            imageSize,
            true
        )

        val input = ByteBuffer.allocateDirect(
            imageSize * imageSize * 3 * 4
        ).order(ByteOrder.nativeOrder())

        for (y in 0 until imageSize) {
            for (x in 0 until imageSize) {

                val pixel = bitmap.getPixel(x, y)

                val red = (pixel shr 16) and 255
                val green = (pixel shr 8) and 255
                val blue = pixel and 255

                input.putFloat(red.toFloat())
                input.putFloat(green.toFloat())
                input.putFloat(blue.toFloat())
            }
        }

        input.rewind()

        val output = Array(1) { FloatArray(1) }

        interpreter.run(input, output)


        val score = output[0][0]

        android.util.Log.d(
            "RETINA_DEBUG",
            "Image size: ${source.width}x${source.height}, " +
                    "Resized: ${bitmap.width}x${bitmap.height}, " +
                    "Retinal score: $score"
        )

        if (bitmap !== source) bitmap.recycle()
        return score
    }

    fun close() {
        interpreter.close()
    }
}
