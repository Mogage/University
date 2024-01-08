package com.example.myapp.todo.ui.items

import SyncJobViewModel
import android.app.Application
import android.util.Log
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.myapp.R
import com.example.myapp.util.createNotificationChannel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ItemsScreen(
    onItemClick: (id: String?) -> Unit,
    onAddItem: () -> Unit,
    onLogout: () -> Unit,
    onOpenMap: () -> Unit,
    isOnline: Boolean
) {
    Log.d("ItemsScreen", "recompose")
    val context = LocalContext.current
    val channelId = "MyTestChannel"

    val myJobsViewModel = viewModel<SyncJobViewModel>(
        factory = SyncJobViewModel.Factory(
            LocalContext.current.applicationContext as Application
        )
    )

    LaunchedEffect(isOnline) {
        createNotificationChannel(channelId, context)
        if (isOnline) {
            myJobsViewModel.enqueueWorker()
        } else {
            myJobsViewModel.cancelWorker()
        }
    }

    val itemsViewModel = viewModel<ItemsViewModel>(factory = ItemsViewModel.Factory)
    val itemsUiState by itemsViewModel.uiState.collectAsStateWithLifecycle(
        initialValue = listOf()
    )
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(text = stringResource(id = R.string.items)) },
                actions = {
                    Button(onClick = onOpenMap) { Text("Map") }
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
                    Log.d("ItemsScreen", "add")
                    onAddItem()
                },
            ) { Icon(Icons.Rounded.Add, "Add") }
        }
    ) {
        ItemList(
            itemList = itemsUiState,
            onItemClick = onItemClick,
            modifier = Modifier.padding(it)
        )
    }
}

//@Preview
//@Composable
//fun PreviewItemsScreen() {
//    ItemsScreen(onItemClick = {}, onAddItem = {}, onLogout = {})
//}
