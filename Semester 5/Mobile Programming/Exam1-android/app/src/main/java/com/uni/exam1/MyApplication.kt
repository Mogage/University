package com.uni.exam1

import android.app.Application
import android.util.Log
import com.uni.exam1.core.TAG

class MyApplication : Application() {
    lateinit var container: AppContainer

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "init")
        container = AppContainer(this)
    }
}
