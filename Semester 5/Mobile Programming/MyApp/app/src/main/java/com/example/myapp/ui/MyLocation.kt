package com.example.myapp.ui

import android.app.Application
import androidx.compose.material.LinearProgressIndicator
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun MyLocation(onExitMap: () -> Unit) {
    val myNetworkStatusViewModel = viewModel<MyLocationViewModel>(
        factory = MyLocationViewModel.Factory(
            LocalContext.current.applicationContext as Application
        )
    )
    val location = myNetworkStatusViewModel.uiState
    if (location != null) {
        MyMap(location.latitude, location.longitude, onExitMap)
    } else {
        LinearProgressIndicator()
    }
}
