package com.example.myapp.todo.ui

import android.util.Log
import android.widget.DatePicker
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.myapp.R
import com.example.myapp.core.Result
import com.example.myapp.todo.ui.item.ItemViewModel
import com.example.myapp.util.showSimpleNotificationWithTapAction

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ItemScreen(itemId: String?, onClose: () -> Unit, isOnline: Boolean) {
    val context = LocalContext.current
    val channelId = "MyTestChannel"
    val itemViewModel = viewModel<ItemViewModel>(factory = ItemViewModel.Factory(itemId))
    val itemUiState = itemViewModel.uiState
    var title by rememberSaveable { mutableStateOf(itemUiState.item.title) }
    var type by rememberSaveable { mutableStateOf(itemUiState.item.type) }
    var noOfGuests by rememberSaveable { mutableIntStateOf(itemUiState.item.noOfGuests) }
    var startDate by rememberSaveable { mutableStateOf(itemUiState.item.startDate) }
    var endDate by rememberSaveable { mutableStateOf(itemUiState.item.endDate) }
    var isCompleted by rememberSaveable { mutableStateOf(itemUiState.item.isCompleted) }
    var doesRepeat by rememberSaveable { mutableStateOf(itemUiState.item.doesRepeat) }
    Log.d("ItemScreen", "recompose, title = $title")

    LaunchedEffect(itemUiState.submitResult) {
        Log.d("ItemScreen", "Submit = ${itemUiState.submitResult}");
        if (itemUiState.submitResult is Result.Success) {
            Log.d("ItemScreen", "Closing screen");
            onClose();
        }
    }

    var textInitialized by remember { mutableStateOf(itemId == null) }
    LaunchedEffect(itemId, itemUiState.loadResult) {
        Log.d("ItemScreen", "Text initialized = ${itemUiState.loadResult}");
        if (textInitialized) {
            return@LaunchedEffect
        }
        if (!(itemUiState.loadResult is Result.Loading)) {
            title = itemUiState.item.title
            type = itemUiState.item.type
            noOfGuests = itemUiState.item.noOfGuests
            startDate = itemUiState.item.startDate
            endDate = itemUiState.item.endDate
            isCompleted = itemUiState.item.isCompleted
            doesRepeat = itemUiState.item.doesRepeat
            textInitialized = true
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(text = stringResource(id = R.string.item)) },
                actions = {
                    Button(onClick = {
                        Log.d("ItemScreen", "save item title = $title")
                        itemViewModel.saveOrUpdateItem(
                            title,
                            type,
                            noOfGuests,
                            startDate,
                            endDate,
                            isCompleted,
                            doesRepeat,
                            isOnline
                        )

                        if (isOnline) {
                            showSimpleNotificationWithTapAction(
                                context = context,
                                channelId = channelId,
                                notificationId = 1,
                                textTitle = "Item saved",
                                textContent = "Item with title $title saved"
                            )
                        }
                        else {
                            showSimpleNotificationWithTapAction(
                                context = context,
                                channelId = channelId,
                                notificationId = 1,
                                textTitle = "Item saved locally",
                                textContent = "Item with title $title saved locally. Sync when online"
                            )
                        }
                    }) { Text("Save") }
                }
            )
        }
    ) {
        Column(
            modifier = Modifier
                .padding(it)
                .fillMaxSize()
        ) {
            if (itemUiState.loadResult is Result.Loading) {
                CircularProgressIndicator()
                return@Scaffold
            }
            if (itemUiState.submitResult is Result.Loading) {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) { LinearProgressIndicator() }
            }
            if (itemUiState.loadResult is Result.Error) {
                Text(text = "Failed to load item - ${(itemUiState.loadResult as Result.Error).exception?.message}")
            }
            Row {
                TextField(
                    value = title,
                    onValueChange = { title = it }, label = { Text("Title") },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            Row {
                TextField(
                    value = type,
                    onValueChange = { type = it }, label = { Text("Type") },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            Row {
                TextField(
                    value = noOfGuests.toString(),
                    onValueChange = { noOfGuests = it.toInt() }, label = { Text("No of Guests") },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            Row {
                TextField(
                    value = startDate,
                    onValueChange = { startDate = it }, label = { Text("Start Date") },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            Row {
                TextField(
                    value = endDate,
                    onValueChange = { endDate = it }, label = { Text("End Date") },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            Row {
                Text(text = "Is completed ")
                Checkbox(
                    checked = isCompleted,
                    onCheckedChange = { isCompleted = it },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            Row {
                Text(text = "Does repeat ")
                Checkbox(
                    checked = doesRepeat,
                    onCheckedChange = { doesRepeat = it },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            if (itemUiState.submitResult is Result.Error) {
                Text(
                    text = "Failed to submit item - ${(itemUiState.submitResult as Result.Error).exception?.message}",
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
    }
}

//
//@Preview
//@Composable
//fun PreviewItemScreen() {
//    ItemScreen(itemId = "0", onClose = {})
//}
