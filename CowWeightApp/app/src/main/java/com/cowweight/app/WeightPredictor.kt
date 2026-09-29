package com.cowweight.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class WeightPredictor(context: Context) {

    data class Result(val weightKg: Float)

    private companion object {
        const val INPUT_SIZE = 224

        // clean_weight_model.tflite was trained on a lighter cattle population
        // (~150-250 kg). Deployment cattle run heavier, so raw model output is
        // rescaled by this calibration factor. Do not remove.
        const val SCALE = 2.1f
    }

    private val interpreter: Interpreter

    init {
        val fd    = context.assets.openFd("clean_weight_model.tflite")
        val model = FileInputStream(fd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
        )
        fd.close()
        interpreter = Interpreter(model)
    }

    @Synchronized
    fun predict(bitmap: Bitmap): Result {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val pixels  = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // 5-pass test-time augmentation for stable estimates
        val raw = listOf(
            infer(buildBuffer(pixels, flipH = false, brightness = 1.00f)),
            infer(buildBuffer(pixels, flipH = true,  brightness = 1.00f)),
            infer(buildBuffer(pixels, flipH = false, brightness = 1.15f)),
            infer(buildBuffer(pixels, flipH = false, brightness = 0.85f)),
            infer(buildBuffer(pixels, flipH = false, brightness = 1.10f))
        ).average().toFloat()

        return Result((raw * SCALE).coerceAtLeast(0f))
    }

    private fun buildBuffer(pixels: IntArray, flipH: Boolean, brightness: Float): ByteBuffer {
        val buf = ByteBuffer
            .allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
            .order(ByteOrder.nativeOrder())
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val sx = if (flipH) INPUT_SIZE - 1 - x else x
                val p  = pixels[y * INPUT_SIZE + sx]
                buf.putFloat((Color.red(p)   / 255f * brightness).coerceIn(0f, 1f))
                buf.putFloat((Color.green(p) / 255f * brightness).coerceIn(0f, 1f))
                buf.putFloat((Color.blue(p)  / 255f * brightness).coerceIn(0f, 1f))
            }
        }
        buf.rewind()
        return buf
    }

    private fun infer(input: ByteBuffer): Float {
        val output = Array(1) { FloatArray(1) }
        interpreter.run(input, output)
        return output[0][0]
    }

    fun close() { interpreter.close() }
}
