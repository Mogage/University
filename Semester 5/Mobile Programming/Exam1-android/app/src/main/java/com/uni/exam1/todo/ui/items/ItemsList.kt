package com.uni.exam1.todo.ui.items

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import com.uni.exam1.todo.data.Item
import kotlinx.coroutines.delay

@Composable
fun ItemsList(itemList: List<Item>, modifier: Modifier) {
    LazyColumn(
        modifier = modifier
            .fillMaxSize()
    ) {
        items(itemList) { item ->
            Row {
                Text(text = "Question ${itemList.indexOf(item)} / ${itemList.size}", modifier = Modifier.fillMaxWidth())
            }
            ItemDetail(item)
        }
    }
}

@Composable
fun ItemDetail(item: Item) {
    val optionStates = remember { mutableStateListOf<Boolean>() }

    Row{
        Text(text = item.text, modifier = Modifier.fillMaxWidth())
    }

    item.options.forEachIndexed { index, option ->
        val isSelected = optionStates.getOrNull(index) ?: false

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(if (isSelected) Color.Red else Color.Transparent)
                .clickable {
                    optionStates.clear()
                    optionStates.addAll(List(item.options.size) { i -> i == index })
                }
        ) {
            Text(text = option)
        }
    }
}