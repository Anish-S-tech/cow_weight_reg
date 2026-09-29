package com.cowweight.app

import android.app.Application
import android.util.Log

class CowWeightApp : Application() {

    @Volatile var predictor: WeightPredictor? = null
        private set
    @Volatile var detector: CowDetector? = null
        private set
    @Volatile var modelsReady = false
        private set
    @Volatile var loadError: String? = null
        private set

    override fun onCreate() {
        super.onCreate()
        Thread {
            try {
                predictor   = WeightPredictor(this)
                detector    = CowDetector(this)
                modelsReady = true
            } catch (e: Exception) {
                loadError = "Could not load AI models: ${e.message}"
                Log.e("CowWeightApp", "Model init failed", e)
            }
        }.apply { name = "model-init"; isDaemon = true }.start()
    }
}
