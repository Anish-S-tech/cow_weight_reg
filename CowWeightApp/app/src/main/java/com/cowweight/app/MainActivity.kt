package com.cowweight.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.cardview.widget.CardView
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface

class MainActivity : AppCompatActivity() {

    private lateinit var btnLiveCamera: Button
    private lateinit var ivPreview:     ImageView
    private lateinit var tvHint:        TextView
    private lateinit var btnGallery:    Button
    private lateinit var btnAnalyze:    Button
    private lateinit var progressBar:   ProgressBar
    private lateinit var cardResult:    CardView
    private lateinit var tvWeight:      TextView
    private lateinit var cardWarning:   CardView
    private lateinit var tvWarnTitle:   TextView
    private lateinit var tvWarnMsg:     TextView

    private var currentBitmap: Bitmap? = null

    // ── Launchers ─────────────────────────────────────────────────────────────

    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let {
            val bmp = decodeSampled(it)
            if (bmp != null) showPreview(correctOrientation(bmp, it))
            else toast("Could not load image — please try again")
        }
    }

    private val startupPermLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { /* handled lazily */ }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnLiveCamera = findViewById(R.id.btnLiveCamera)
        ivPreview     = findViewById(R.id.ivPreview)
        tvHint        = findViewById(R.id.tvHint)
        btnGallery    = findViewById(R.id.btnGallery)
        btnAnalyze    = findViewById(R.id.btnAnalyze)
        progressBar   = findViewById(R.id.progressBar)
        cardResult    = findViewById(R.id.cardResult)
        tvWeight      = findViewById(R.id.tvWeight)
        cardWarning   = findViewById(R.id.cardWarning)
        tvWarnTitle   = findViewById(R.id.tvWarnTitle)
        tvWarnMsg     = findViewById(R.id.tvWarnMsg)

        requestStartupPermissions()

        btnLiveCamera.setOnClickListener {
            startActivity(Intent(this, LiveCameraActivity::class.java))
        }
        btnGallery.setOnClickListener { galleryLauncher.launch("image/*") }
        btnAnalyze.setOnClickListener  { currentBitmap?.let { runAnalysis(it) } }
    }

    // ── Permissions ───────────────────────────────────────────────────────────

    private fun requestStartupPermissions() {
        val needed = mutableListOf<String>()
        if (!hasPermission(Manifest.permission.CAMERA))
            needed += Manifest.permission.CAMERA
        val storage = if (android.os.Build.VERSION.SDK_INT >= 33)
            Manifest.permission.READ_MEDIA_IMAGES
        else
            Manifest.permission.READ_EXTERNAL_STORAGE
        if (!hasPermission(storage)) needed += storage
        if (needed.isNotEmpty()) startupPermLauncher.launch(needed.toTypedArray())
    }

    private fun hasPermission(p: String) =
        ContextCompat.checkSelfPermission(this, p) == PackageManager.PERMISSION_GRANTED

    // ── Image loading ─────────────────────────────────────────────────────────

    private fun decodeSampled(uri: Uri): Bitmap? = try {
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }

        var s = 1
        while (opts.outWidth / s > 1024 || opts.outHeight / s > 1024) s *= 2
        opts.inSampleSize       = s
        opts.inJustDecodeBounds = false

        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
    } catch (e: Exception) { null }

    private fun correctOrientation(bmp: Bitmap, uri: Uri): Bitmap = try {
        contentResolver.openInputStream(uri)?.use { stream ->
            val exif = ExifInterface(stream)
            val deg  = when (exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
            )) {
                ExifInterface.ORIENTATION_ROTATE_90  ->  90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> return bmp
            }
            Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height,
                Matrix().apply { postRotate(deg) }, true)
        } ?: bmp
    } catch (e: Exception) { bmp }

    private fun showPreview(bmp: Bitmap) {
        currentBitmap = bmp
        ivPreview.setImageBitmap(bmp)
        ivPreview.scaleType     = ImageView.ScaleType.CENTER_CROP
        tvHint.visibility       = View.GONE
        btnAnalyze.visibility   = View.VISIBLE
        cardResult.visibility   = View.GONE
        cardWarning.visibility  = View.GONE
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    private fun runAnalysis(bitmap: Bitmap) {
        val app = application as CowWeightApp
        if (!app.modelsReady) {
            toast(app.loadError ?: "AI is still loading — please wait a moment")
            return
        }
        val predictor = app.predictor ?: return
        val detector  = app.detector  ?: return

        btnAnalyze.isEnabled    = false
        progressBar.visibility  = View.VISIBLE
        cardResult.visibility   = View.GONE
        cardWarning.visibility  = View.GONE

        Thread {
            try {
                when (detector.detect(bitmap)) {
                    CowDetector.CountResult.NONE ->
                        post { showWarning(
                            "No cow detected",
                            "Please take a clear, well-lit photo with the cow fully visible in the frame."
                        ) }
                    CowDetector.CountResult.MANY ->
                        post { showWarning(
                            "Multiple cows detected",
                            "For accurate weight estimation, please capture only one cow at a time."
                        ) }
                    CowDetector.CountResult.ONE -> {
                        val result = predictor.predict(bitmap)
                        post { showResult(result) }
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("MainActivity", "Analysis failed", e)
                post { showWarning("Could not analyze image", "Please try with a clearer photo.") }
            } finally {
                post {
                    progressBar.visibility = View.GONE
                    btnAnalyze.isEnabled   = true
                }
            }
        }.start()
    }

    private fun post(action: () -> Unit) {
        if (!isDestroyed && !isFinishing) runOnUiThread(action)
    }

    // ── UI ────────────────────────────────────────────────────────────────────

    private fun showResult(result: WeightPredictor.Result) {
        tvWeight.text           = "${result.weightKg.toInt()} kg"
        cardResult.visibility   = View.VISIBLE
        cardWarning.visibility  = View.GONE
    }

    private fun showWarning(title: String, msg: String) {
        tvWarnTitle.text        = title
        tvWarnMsg.text          = msg
        cardWarning.visibility  = View.VISIBLE
        cardResult.visibility   = View.GONE
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
}
