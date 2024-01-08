package com.example.myapp.util

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.example.myapp.AppContainer
import com.example.myapp.MyApplication
import com.example.myapp.auth.TAG

class SyncWorker(appContext: Context, params: WorkerParameters) :
    CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result {
        val appContainer = getAppContainer(applicationContext)
        val itemRepository = appContainer.itemRepository

        Log.d(TAG, "Started syncing: fetching")
        itemRepository.sync()
        itemRepository.refresh()
        Log.d(TAG, "Finished syncing: fetching")

        return Result.success()
    }
    private fun getAppContainer(context: Context): AppContainer {
        if (context.applicationContext is MyApplication) {
            val application = context.applicationContext as MyApplication
            return application.container
        }
        throw IllegalStateException("Invalid context type: Not an application context")
    }
}
