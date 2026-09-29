package com.cowweight.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import java.nio.FloatBuffer

/** Gates weight estimation on frame content — rejects frames with zero or multiple cows. */
class CowDetector(context: Context) {

    private companion object {
        const val MODEL_FILE  = "yolov8n.onnx"
        const val INPUT_SIZE  = 320   // this asset was exported at imgsz=320, not the standard 640
        const val COW_CLASS   = 19    // COCO class index for "cow"
        const val CONF_THRESH = 0.30f
        const val IOU_THRESH  = 0.45f
    }

    enum class CountResult { NONE, ONE, MANY }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    init {
        val bytes = context.assets.open(MODEL_FILE).readBytes()
        session   = env.createSession(bytes, OrtSession.SessionOptions())
        inputName = session.inputNames.iterator().next()
    }

    @Synchronized
    fun detect(bitmap: Bitmap): CountResult {
        val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // CHW float buffer — R plane, then G, then B; values 0-1
        val buf = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
        for (p in pixels) buf.put(Color.red(p)   / 255f)
        for (p in pixels) buf.put(Color.green(p) / 255f)
        for (p in pixels) buf.put(Color.blue(p)  / 255f)
        buf.rewind()

        val tensor = OnnxTensor.createTensor(
            env, buf, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )

        val ortResult = session.run(mapOf(inputName to tensor))
        tensor.close()

        // YOLOv8 output shape: [1, 84, numPredictions]
        // 84 = cx, cy, w, h + 80 class scores (no objectness in v8)
        @Suppress("UNCHECKED_CAST")
        val raw = (ortResult[0].value as Array<Array<FloatArray>>)[0]
        ortResult.close()

        val numPred  = raw[0].size
        val cowBoxes = ArrayList<FloatArray>(64)

        for (i in 0 until numPred) {
            val conf = raw[4 + COW_CLASS][i]
            if (conf < CONF_THRESH) continue
            val cx = raw[0][i]; val cy = raw[1][i]
            val w  = raw[2][i]; val h  = raw[3][i]
            cowBoxes += floatArrayOf(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f, conf)
        }

        val kept = nms(cowBoxes)
        return when {
            kept.isEmpty() -> CountResult.NONE
            kept.size == 1 -> CountResult.ONE
            else           -> CountResult.MANY
        }
    }

    private fun nms(boxes: List<FloatArray>): List<FloatArray> {
        val sorted   = boxes.sortedByDescending { it[4] }
        val suppress = BooleanArray(sorted.size)
        val kept     = ArrayList<FloatArray>()
        for (i in sorted.indices) {
            if (suppress[i]) continue
            kept += sorted[i]
            for (j in i + 1 until sorted.size) {
                if (!suppress[j] && iou(sorted[i], sorted[j]) > IOU_THRESH) suppress[j] = true
            }
        }
        return kept
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val ix = maxOf(0f, minOf(a[2], b[2]) - maxOf(a[0], b[0]))
        val iy = maxOf(0f, minOf(a[3], b[3]) - maxOf(a[1], b[1]))
        val inter = ix * iy
        val union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun close() { session.close() }
}
