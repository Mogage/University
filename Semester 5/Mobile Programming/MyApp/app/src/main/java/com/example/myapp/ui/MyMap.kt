package com.example.myapp.ui

import android.util.Log
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.google.android.gms.maps.model.CameraPosition
import com.google.android.gms.maps.model.LatLng
import com.google.maps.android.compose.*

val TAG = "MyMap"

@Composable
fun MyMap(lat: Double, long: Double, onExitMap: () -> Unit) {
    val markerState = rememberMarkerState(position = LatLng(lat, long))
    val cameraPositionState = rememberCameraPositionState {
        position = CameraPosition.fromLatLngZoom(markerState.position, 10f)
    }

    Column {
        GoogleMap(
            modifier = Modifier.fillMaxSize().weight(1f),
            cameraPositionState = cameraPositionState,
            onMapClick = {
                Log.d(TAG, "onMapClick $it")
            },
            onMapLongClick = {
                Log.d(TAG, "onMapLongClick $it")
                markerState.position = it
            },
        ) {
            Marker(
                state = MarkerState(position = markerState.position),
                title = "User location title",
                snippet = "User location",
            )
        }
        FloatingActionButton(
            modifier = Modifier.fillMaxWidth(),
            onClick = {

                onExitMap()
            },
        ) {
            Icon(Icons.Filled.ArrowBack, "Exit")
        }
    }
}
