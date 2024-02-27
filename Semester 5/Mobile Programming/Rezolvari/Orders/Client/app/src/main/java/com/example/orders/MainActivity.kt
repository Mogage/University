package com.example.orders

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.lifecycleScope
import com.example.orders.core.TAG
import com.example.orders.core.ui.MyNetworkStatus
import com.example.orders.core.utils.createNotificationChannel
import com.example.orders.ui.theme.OrderStoreAndroidTheme
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalPermissionsApi::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            createNotificationChannel(channelId = "Orders Channel",context = this@MainActivity)
            (application as OrderStoreAndroid).container.orderRepository.setContext(this@MainActivity)
            Log.d(TAG, "onCreate")

            OrderStoreAndroid {
                OrderStoreAndroidNavHost()
            }
            MyNetworkStatus()
        }
    }

    private fun askPermissions(context: Context) {
        if (!Settings.System.canWrite(context)) {
            val i = Intent(Settings.ACTION_MANAGE_WRITE_SETTINGS);
            context.startActivity(i);
        }
    }

    override fun onResume() {
        super.onResume()
        lifecycleScope.launch {
            (application as OrderStoreAndroid).container.orderRepository.openWsClient()
            (application as OrderStoreAndroid).container.orderRepository.setContext(this@MainActivity)
        }
    }

    override fun onPause() {
        super.onPause()
        lifecycleScope.launch {
            (application as OrderStoreAndroid).container.orderRepository.closeWsClient()
            (application as OrderStoreAndroid).container.orderRepository.setContext(this@MainActivity)
        }
    }
}

@Composable
fun OrderStoreAndroid(content: @Composable () -> Unit){
    Log.d("OrderStoreAndroid", "recompose")
    OrderStoreAndroidTheme {
        Surface {
            content()
        }
    }
}

@Preview
@Composable
fun PreviewMyApp() {
    OrderStoreAndroid {
        OrderStoreAndroidNavHost()
    }
}
