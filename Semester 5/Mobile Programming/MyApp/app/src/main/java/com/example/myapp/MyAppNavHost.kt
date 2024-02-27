package com.example.myapp

import android.Manifest
import android.app.Activity
import android.content.pm.ActivityInfo
import android.util.Log
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.example.myapp.auth.LoginScreen
import com.example.myapp.core.data.UserPreferences
import com.example.myapp.core.data.remote.Api
import com.example.myapp.core.ui.UserPreferencesViewModel
import com.example.myapp.todo.ui.ItemScreen
import com.example.myapp.todo.ui.OrientationSensorMonitor
import com.example.myapp.todo.ui.items.ItemsScreen
import com.example.myapp.ui.MyLocation
import com.example.myapp.util.ConnectivityManagerNetworkMonitor
import com.example.myapp.util.Permissions
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import kotlinx.coroutines.delay

val itemsRoute = "items"
val authRoute = "auth"
val mapRoute = "map"

@Composable
fun MyAppNavHost() {
    val context = LocalContext.current
    val connectivityManager = remember {
        ConnectivityManagerNetworkMonitor(context)
    }
    val isOnline by connectivityManager.isOnline.collectAsStateWithLifecycle(
        initialValue = false
    )

    val orientationSensorMonitor = OrientationSensorMonitor(context = LocalContext.current)
    val orientation by orientationSensorMonitor.orientation.collectAsStateWithLifecycle(
        initialValue = "portrait"
    )

    var lastOrientation by remember { mutableStateOf("portrait") }
    LaunchedEffect(orientation) {
        if (orientation != lastOrientation) {
            delay(500L)
            when (orientation) {
                "landscape" -> {
                    Log.d("ItemsScreen", "landscape")
                    (context as? Activity)?.requestedOrientation =
                        ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
                }

                "portrait" -> {
                    Log.d("ItemsScreen", "portrait")
                    (context as? Activity)?.requestedOrientation =
                        ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
                }

                "rlandscape" -> {
                    Log.d("ItemsScreen", "rlandscape")
                    (context as? Activity)?.requestedOrientation =
                        ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE
                }

                "rportrait" -> {
                    Log.d("ItemsScreen", "rportrait")
                    (context as? Activity)?.requestedOrientation =
                        ActivityInfo.SCREEN_ORIENTATION_REVERSE_PORTRAIT
                }

                else -> {
                    Log.d("ItemsScreen", "portrait")
                    (context as? Activity)?.requestedOrientation =
                        ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
                }
            }
            lastOrientation = orientation
        }
    }

    val navController = rememberNavController()
    val onCloseItem = {
        Log.d("MyAppNavHost", "navigate back to list")
        navController.popBackStack()
    }
    val onExitMap = {
        Log.d("MyAppNavHost", "navigate back to list")
        navController.popBackStack()
    }
    val userPreferencesViewModel =
        viewModel<UserPreferencesViewModel>(factory = UserPreferencesViewModel.Factory)
    val userPreferencesUiState by userPreferencesViewModel.uiState.collectAsStateWithLifecycle(
        initialValue = UserPreferences()
    )

    val myAppViewModel = viewModel<MyAppViewModel>(factory = MyAppViewModel.Factory)
    NavHost(
        navController = navController,
        startDestination = authRoute
    ) {
        composable(itemsRoute) {
            ItemsScreen(
                onItemClick = { itemId ->
                    Log.d("MyAppNavHost", "navigate to item $itemId")
                    navController.navigate("$itemsRoute/$itemId")
                },
                onAddItem = {
                    Log.d("MyAppNavHost", "navigate to new item")
                    navController.navigate("$itemsRoute-new")
                },
                onLogout = {
                    Log.d("MyAppNavHost", "logout")
                    myAppViewModel.logout()
                    Api.tokenInterceptor.token = null
                    navController.navigate(authRoute) {
                        popUpTo(0)
                    }
                },
                onOpenMap = {
                    Log.d("MyAppNavHost", "navigate to map")
                    navController.navigate("map")
                },
                isOnline = isOnline
            )
        }
        composable(
            route = "$itemsRoute/{id}",
            arguments = listOf(navArgument("id") { type = NavType.StringType })
        )
        {
            ItemScreen(
                itemId = it.arguments?.getString("id"),
                isOnline = isOnline,
                onClose = { onCloseItem() }
            )
        }
        composable(route = "$itemsRoute-new")
        {
            ItemScreen(
                itemId = null,
                isOnline = isOnline,
                onClose = { onCloseItem() }
            )
        }
        composable(route = authRoute)
        {
            LoginScreen(
                onClose = {
                    Log.d("MyAppNavHost", "navigate to list")
                    navController.navigate(itemsRoute)
                }
            )
        }
        composable(route = mapRoute)
        {
            MapContent(
                onExitMap = { onExitMap() }
            )
        }
    }
    LaunchedEffect(userPreferencesUiState.token) {
        if (userPreferencesUiState.token.isNotEmpty()) {
            Log.d("MyAppNavHost", "Lauched effect navigate to items")
            Api.tokenInterceptor.token = userPreferencesUiState.token
            myAppViewModel.setToken(userPreferencesUiState.token)
            navController.navigate(itemsRoute) {
                popUpTo(0)
            }
        }
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun MapContent(onExitMap: () -> Unit) {
    Permissions(
        permissions = listOf(
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.ACCESS_FINE_LOCATION
        ),
        rationaleText = "Please allow app to use location (coarse or fine)",
        dismissedText = "O noes! No location provider allowed!"
    ) {
        MyLocation(onExitMap)
    }
}
