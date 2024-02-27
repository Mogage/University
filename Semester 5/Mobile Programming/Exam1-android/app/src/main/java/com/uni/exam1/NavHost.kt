package com.uni.exam1

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
import com.uni.exam1.auth.LoginScreen
import com.uni.exam1.core.data.UserPreferences
import com.uni.exam1.core.data.remote.Api
import com.uni.exam1.core.ui.UserPreferencesViewModel
import com.uni.exam1.todo.ui.QuestionIdsViewModel
import com.uni.exam1.todo.ui.item.ItemScreen
import com.uni.exam1.todo.ui.items.ItemsScreen
import com.uni.exam1.util.ConnectivityManagerNetworkMonitor

const val itemsRoute = "question"
const val authRoute = "auth"

@Composable
fun NavHost() {
    val context = LocalContext.current
    val connectivityManager = remember {
        ConnectivityManagerNetworkMonitor(context)
    }
    val isOnline by connectivityManager.isOnline.collectAsStateWithLifecycle(
        initialValue = false
    )

    val navController = rememberNavController()
    val userPreferencesViewModel =
        viewModel<UserPreferencesViewModel>(factory = UserPreferencesViewModel.Factory)
    val userPreferencesUiState by userPreferencesViewModel.uiState.collectAsStateWithLifecycle(
        initialValue = UserPreferences()
    )

    val exam1ViewModel = viewModel<Exam1ViewModel>(factory = Exam1ViewModel.Factory)

    NavHost(
        navController = navController,
        startDestination = authRoute
    ) {
        composable(
            route = "$itemsRoute/{id}",
            arguments = listOf(navArgument("id") { type = NavType.StringType })
        )
        {
            ItemsScreen(
                onItemClick = { itemId ->
                    Log.d("MyAppNavHost", "to highlight item $itemId")
                    // TODO: highlight item
                },
                onLogout = {
                    Log.d("MyAppNavHost", "logout")
                    exam1ViewModel.logout()
                    Api.tokenInterceptor.token = null
                    navController.navigate(authRoute) {
                        popUpTo(0)
                    }
                },
                onNextItem = { itemId ->
                    Log.d("MyAppNavHost", "navigate to item $itemId")
                    navController.navigate("$itemsRoute/$itemId")
                },
                isOnline = isOnline
            )
        }
        composable(route = authRoute)
        {
            LoginScreen(
                onClose = {
                    Log.d("MyAppNavHost", "navigate to list")
                    navController.navigate("$itemsRoute/0")
                }
            )
        }
    }
//    LaunchedEffect(userPreferencesUiState.token) {
//        if (userPreferencesUiState.token.isNotEmpty()) {
//            Log.d("MyAppNavHost", "Launched effect navigate to items")
//            Api.tokenInterceptor.token = userPreferencesUiState.token
//            exam1ViewModel.setToken(userPreferencesUiState.token)
//            navController.navigate("$itemsRoute/0") {
//                popUpTo(0)
//            }
//        }
//    }
}
