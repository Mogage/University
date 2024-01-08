import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import androidx.work.*
import com.example.myapp.util.SyncWorker
import kotlinx.coroutines.launch
import java.util.*
import java.util.concurrent.TimeUnit


class SyncJobViewModel(application: Application) : AndroidViewModel(application) {
    private var workManager: WorkManager
    private var workId: UUID? = null

    init {
        workManager = WorkManager.getInstance(getApplication())
        startJob()
    }

    private fun startJob() {
        viewModelScope.launch {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build()
            val myPeriodicWork = PeriodicWorkRequestBuilder<SyncWorker>(
                repeatInterval = 1, // Interval between repeats
                repeatIntervalTimeUnit = TimeUnit.SECONDS
            )
                .setConstraints(constraints)
                .build()

            workManager.apply {
                // enqueue Work
                enqueue(myPeriodicWork)
            }
        }
    }

    fun enqueueWorker() {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .build()
        val myPeriodicWork = PeriodicWorkRequestBuilder<SyncWorker>(
            repeatInterval = 1,
            repeatIntervalTimeUnit = TimeUnit.SECONDS
        )
            .setConstraints(constraints)
            .build()

        workManager.enqueueUniquePeriodicWork(
            "SyncWorker",
            ExistingPeriodicWorkPolicy.REPLACE,
            myPeriodicWork // your PeriodicWorkRequest
        )
    }

    fun cancelWorker() {
        workManager.cancelUniqueWork("SyncWorker")
    }

    companion object {
        fun Factory(application: Application): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                SyncJobViewModel(application)
            }
        }
    }
}
