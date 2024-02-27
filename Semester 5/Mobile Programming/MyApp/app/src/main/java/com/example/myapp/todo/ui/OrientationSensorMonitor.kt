package com.example.myapp.todo.ui

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlin.math.abs

@OptIn(ExperimentalCoroutinesApi::class)
class OrientationSensorMonitor(val context: Context) {
    val orientation: Flow<String> = callbackFlow<String> {
        val sensorManager: SensorManager =
            context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val accelerometerSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        val accelerometerSensorEventListener = object : SensorEventListener {
            override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
                // Not used in this example
            }

            override fun onSensorChanged(event: SensorEvent) {
                if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                    val x = event.values[0]
                    val y = event.values[1]

                    if (abs(x) > abs(y)) {
                        if (x > 0) {
                            channel.trySend("landscape")
                        } else {
                            channel.trySend("rlandscape")
                        }
                    } else {
                        if (y > 0) {
                            channel.trySend("portrait")
                        } else {
                            channel.trySend("rportrait")
                        }
                    }
                }
            }
        }

        sensorManager.registerListener(
            accelerometerSensorEventListener,
            accelerometerSensor,
            SensorManager.SENSOR_DELAY_NORMAL
        )

        awaitClose {
            sensorManager.unregisterListener(accelerometerSensorEventListener)
        }
    }
}
//
//class OrientationSensorMonitor(val context: Context){
//    val isLandscape: Flow<Boolean> = callbackFlow<Boolean> {
//        val sensorManager: SensorManager =
//            context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
//        val humiditySensor = sensorManager.getDefaultSensor(Sensor.TYPE_RELATIVE_HUMIDITY)
//        val humidityListener = object : SensorEventListener {
//            override fun onSensorChanged(event: SensorEvent) {
//                if (event.sensor.type == Sensor.TYPE_RELATIVE_HUMIDITY) {
//                    if (event.values[0] <= 30f) {
//                        channel.trySend(false)
//                    } else {
//                        channel.trySend(true)
//                    }
//                }
//            }
//            override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
//            }
//        }
//        sensorManager.registerListener(
//            humidityListener,
//            humiditySensor,
//            SensorManager.SENSOR_DELAY_NORMAL
//        )
//
//        awaitClose { sensorManager.unregisterListener(humidityListener) }
//    }
//}