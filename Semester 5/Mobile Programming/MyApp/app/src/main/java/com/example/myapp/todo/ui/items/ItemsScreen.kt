package com.example.myapp.todo.ui.items

import MyFloatingActionButton
import SyncJobViewModel
import android.app.Application
import android.util.Log
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.myapp.R
import com.example.myapp.util.createNotificationChannel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

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
    var isEditing by remember { mutableStateOf(false) }
    val coroutineScope = rememberCoroutineScope()
    suspend fun showEditMessage() {
        if (!isEditing) {
            isEditing = true
            delay(1000L)
            isEditing = false
            onAddItem()
        }
    }

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
            MyFloatingActionButton(onClick = {
                coroutineScope.launch {
                    Log.d("ItemsScreen", "add")
                    showEditMessage()
                }
            }, isEditing = isEditing)

//            FloatingActionButton(
//                onClick = {
//                    Log.d("ItemsScreen", "add")
//                    onAddItem()
//                },
//            ) { Icon(Icons.Rounded.Add, "Add") }
        }
    ) {
        ItemList(
            itemList = itemsUiState,
            onItemClick = onItemClick,
            modifier = Modifier.padding(it)
        )
    }
}
