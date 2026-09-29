package com.cowweight.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.cardview.widget.CardView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

class LiveCameraActivity : AppCompatActivity() {

    private companion object {
        // Cadence for the continuous "is a cow in frame" preview check — cheap enough
        // to run repeatedly without competing with the camera pipeline for CPU.
        const val LIVE_DETECT_INTERVAL_MS = 400L
    }

    private lateinit var previewView:  PreviewView
    private lateinit var tvStatus:     TextView
    private lateinit var btnCapture:   Button
    private lateinit var cardResult:   CardView
    private lateinit var tvLiveWeight: TextView
    private lateinit var btnScanAgain: Button
    private lateinit var btnClose:     Button

    private val mainHandler    = Handler(Looper.getMainLooper())
    private val cameraExecutor = Executors.newSingleThreadExecutor()

    // Most-recent camera frame — written on camera executor, read on main thread
    @Volatile private var latestFrame: Bitmap? = null
    @Volatile private var busy = false
    private var lastLiveDetectAt = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_live_camera)

        previewView  = findViewById(R.id.previewView)
        tvStatus     = findViewById(R.id.tvStatus)
        btnCapture   = findViewById(R.id.btnCapture)
        cardResult   = findViewById(R.id.cardResult)
        tvLiveWeight = findViewById(R.id.tvLiveWeight)
        btnScanAgain = findViewById(R.id.btnScanAgain)
        btnClose     = findViewById(R.id.btnClose)

        btnClose.setOnClickListener     { finish() }
        btnScanAgain.setOnClickListener { resetToScanning() }
        btnCapture.setOnClickListener   { captureAndEstimate() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            tvStatus.text        = "Camera permission required"
            btnCapture.isEnabled = false
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacksAndMessages(null)
        cameraExecutor.shutdown()
    }

    // ── Camera ────────────────────────────────────────────────────────────────

    private fun startCamera() {
        ProcessCameraProvider.getInstance(this).addListener({
            val provider = ProcessCameraProvider.getInstance(this).get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { ia ->
                    ia.setAnalyzer(cameraExecutor, ::storeFrame)
                }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
                )
            } catch (e: Exception) {
                tvStatus.text = "Camera error — please restart the app"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun storeFrame(imageProxy: ImageProxy) {
        if (busy) { imageProxy.close(); return }

        val bitmap = try {
            correctedBitmap(imageProxy)
        } catch (e: Exception) {
            Log.e("LiveCameraActivity", "Frame decode failed", e)
            null
        } finally {
            imageProxy.close()
        }
        if (bitmap == null) return
        latestFrame = bitmap

        val now = System.currentTimeMillis()
        if (now - lastLiveDetectAt < LIVE_DETECT_INTERVAL_MS) return
        lastLiveDetectAt = now
        runLiveDetectPreview(bitmap)
    }

    // CameraX delivers frames in the sensor's native orientation — imageInfo.rotationDegrees
    // is the rotation needed to make them upright. Without this, both YOLO and the weight
    // model see a sideways frame on a phone held in portrait, which quietly wrecks accuracy
    // instead of throwing.
    private fun correctedBitmap(imageProxy: ImageProxy): Bitmap {
        val bitmap = imageProxy.toBitmap()
        val rotation = imageProxy.imageInfo.rotationDegrees
        if (rotation == 0) return bitmap
        return Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height,
            Matrix().apply { postRotate(rotation.toFloat()) }, true
        )
    }

    /** Lightweight, throttled detection-only pass so the status text reacts live as the user aims. */
    private fun runLiveDetectPreview(frame: Bitmap) {
        val app = application as CowWeightApp
        if (!app.modelsReady) return
        val detector = app.detector ?: return

        try {
            val result = detector.detect(frame)
            mainHandler.post { updateLiveStatus(result) }
        } catch (e: Exception) {
            Log.e("LiveCameraActivity", "Live preview detect failed", e)
        }
    }

    private fun updateLiveStatus(result: CowDetector.CountResult) {
        if (busy || isDestroyed || isFinishing || cardResult.visibility == View.VISIBLE) return
        tvStatus.text = when (result) {
            CowDetector.CountResult.NONE -> "Point camera at a cow"
            CowDetector.CountResult.ONE  -> "Cow detected — tap to weigh"
            CowDetector.CountResult.MANY -> "Multiple cows — frame just one"
        }
    }

    // ── Capture ───────────────────────────────────────────────────────────────

    private fun captureAndEstimate() {
        val frame = latestFrame
        if (frame == null) {
            tvStatus.text = "Camera not ready — please wait"
            return
        }

        val app = application as CowWeightApp
        if (!app.modelsReady) {
            tvStatus.text = app.loadError ?: "AI is still loading — please wait"
            return
        }
        val predictor = app.predictor ?: return
        val detector  = app.detector  ?: return

        busy                  = true
        btnCapture.isEnabled  = false
        tvStatus.text         = "Detecting cow…"

        Thread {
            try {
                when (detector.detect(frame)) {
                    CowDetector.CountResult.NONE -> mainHandler.post {
                        if (isDestroyed || isFinishing) return@post
                        tvStatus.text         = "No cow detected — adjust the frame and try again"
                        btnCapture.isEnabled  = true
                        busy                  = false
                    }
                    CowDetector.CountResult.MANY -> mainHandler.post {
                        if (isDestroyed || isFinishing) return@post
                        tvStatus.text         = "Multiple cows visible — frame a single cow and try again"
                        btnCapture.isEnabled  = true
                        busy                  = false
                    }
                    CowDetector.CountResult.ONE -> {
                        mainHandler.post {
                            if (isDestroyed || isFinishing) return@post
                            tvStatus.text = "Estimating weight…"
                        }
                        val result = predictor.predict(frame)
                        mainHandler.post {
                            if (isDestroyed || isFinishing) return@post
                            showResult(result)
                        }
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("LiveCameraActivity", "Analysis failed", e)
                mainHandler.post {
                    if (isDestroyed || isFinishing) return@post
                    tvStatus.text         = "Could not analyze — please try again"
                    btnCapture.isEnabled  = true
                    busy                  = false
                }
            }
        }.start()
    }

    // ── UI ────────────────────────────────────────────────────────────────────

    private fun showResult(result: WeightPredictor.Result) {
        tvLiveWeight.text        = "${result.weightKg.toInt()} kg"
        tvStatus.text            = "Estimation complete"
        cardResult.visibility    = View.VISIBLE
        btnScanAgain.visibility  = View.VISIBLE
        btnCapture.visibility    = View.GONE
    }

    private fun resetToScanning() {
        busy                     = false
        latestFrame              = null
        cardResult.visibility    = View.GONE
        btnScanAgain.visibility  = View.GONE
        btnCapture.visibility    = View.VISIBLE
        btnCapture.isEnabled     = true
        tvStatus.text            = "Point camera at a cow"
    }
}
