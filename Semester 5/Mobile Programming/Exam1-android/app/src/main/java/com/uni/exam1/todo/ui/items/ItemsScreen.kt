package com.uni.exam1.todo.ui.items

import android.util.Log
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ArrowForward
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.uni.exam1.R
import com.uni.exam1.todo.ui.item.ItemScreen


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ItemsScreen(
    onItemClick: (id: String?) -> Unit,
    onLogout: () -> Unit,
    onNextItem: (id: String?) -> Unit,
    isOnline: Boolean
) {
    Log.d("ItemsScreen", "recompose")
    val context = LocalContext.current
    val channelId = "MyTestChannel"

    val itemsViewModel = viewModel<ItemsViewModel>(factory = ItemsViewModel.Factory)
    val itemsUiState by itemsViewModel.uiState.collectAsStateWithLifecycle(
        initialValue = listOf()
    )

//    val itemViewModel = viewModel<ItemViewModel>(factory = ItemViewModel.Factory("0"))
//    val itemUiState = itemViewModel.uiState
//    val text by rememberSaveable { mutableStateOf(itemUiState.item.text) }
//    val options by rememberSaveable { mutableStateOf(itemUiState.item.options) }
//    val indexCorrectOption by rememberSaveable { mutableIntStateOf(itemUiState.item.indexCorrectOption) }
//    Log.d("ItemScreen", "recompose, text = $text")

//    val myJobsViewModel = viewModel<SyncJobViewModel>(
//        factory = SyncJobViewModel.Factory(
//            LocalContext.current.applicationContext as Application
//        )
//    )
//
//    LaunchedEffect(isOnline) {
//        createNotificationChannel(channelId, context)
//        if (isOnline) {
//            myJobsViewModel.enqueueWorker()
//        } else {
//            myJobsViewModel.cancelWorker()
//        }
//    }
//
//    val itemsViewModel = viewModel<ItemViewModel>(factory = ItemViewModel.Factory)
//    val itemsUiState by itemsViewModel.uiState.collectAsStateWithLifecycle(
//        initialValue = listOf()
//    )

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(text = stringResource(id = R.string.items)) },
                actions = {
                    Button(onClick = onLogout) { Text("Logout") }
                }
            )
        },
        bottomBar = {
            if (isOnline) {
                Text(text = "Online")
            } else {
                Text(text = "Offline")
            }
        },
        floatingActionButton = {
            FloatingActionButton(
                onClick = {
                    Log.d("ItemsScreen", "next")
                    onNextItem("2")
                },
            ) { Icon(Icons.Rounded.ArrowForward, "Next") }
        }
    ) {
        ItemsList(
            itemList = itemsUiState,
            modifier = Modifier.padding(it)
        )
    }
}

