package com.example.items.todo.ui.items


import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.CornerSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.items.todo.data.Item
import kotlinx.coroutines.delay

typealias OnItemFn = (id: Int?) -> Unit

@Composable
fun ItemList(itemList: List<Item>, onItemClick: OnItemFn, modifier: Modifier) {
    Log.d("ItemList", "recompose")

    var currentItemIndex by remember { mutableIntStateOf(0) }
    var correctAnswers by remember { mutableIntStateOf(0) }
    var totalAnsweredItems by remember { mutableIntStateOf(0) }
    var awaitingSubmit by remember { mutableStateOf(true) }

    ItemHeader(
        totalAnsweredItems = totalAnsweredItems,
        totalItems = itemList.size,
        correctAnswers = correctAnswers
    )

    if (currentItemIndex < itemList.size) {
        // Show the current item
        Column(
            modifier = Modifier.fillMaxSize()
        ) {
            // Display the header with information about answered items and correct answers

            Text(text = "\n\n")
            // Show the current item
            ItemDetail(itemList[currentItemIndex], onAnswerSubmit =
            {
                    answer ->
                run {
                    Log.d("ItemList", "Answer submitted is correct: $answer")
                    totalAnsweredItems++
                    if (answer != false) {
                        correctAnswers++
                    }
                    // Set the flag to indicate that submit was pressed
                    awaitingSubmit = false
                    // Increment currentItemIndex after the answer is submitted
                    currentItemIndex++
                }
            })
        }

        // Use a LaunchedEffect to automatically move to the next item after 10 seconds
        LaunchedEffect(currentItemIndex, awaitingSubmit) {
            Log.d("ItemList", "LaunchedEffect start waiting for 10 seconds")
            delay(10000)
            // Check if submit was not pressed during the 10 seconds
            if (awaitingSubmit) {
                currentItemIndex++
                totalAnsweredItems++
            }
            else
            {
                awaitingSubmit = true
            }
            Log.d("ItemList", "LaunchedEffect end waiting for 10 seconds")
        }
    } else {
        // All items answered
        Text("All items answered")
    }
}


@Composable
fun ItemHeader(
    totalAnsweredItems: Int,
    totalItems: Int,
    correctAnswers: Int
) {
    // Display the header with information about answered items and correct answers
    Text(
        text = "\n\n\n\t\tItems $totalAnsweredItems/$totalItems, Correct answers: $correctAnswers/$totalAnsweredItems",
        style = MaterialTheme.typography.bodyMedium
    )
}

@Composable
fun ItemDetail(
    item: Item,
    onAnswerSubmit: (answer : Boolean?) -> Unit
) {

    var selectedIndex by remember { mutableStateOf(item.selectedIndex) }
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(46.dp)
            .clip(
                shape = RoundedCornerShape(
                    topStart = CornerSize(5.dp),
                    topEnd = CornerSize(30.dp),
                    bottomStart = CornerSize(30.dp),
                    bottomEnd = CornerSize(5.dp)
                )
            )
            .background(color = Color.White)
            .padding(12.dp)
    ) {
        Column {
            Row {
                Text(
                    text = "Text: ${item.text}",
                    style = MaterialTheme.typography.bodyLarge.copy(
                        color = Color.Black
                    ),
                    modifier = Modifier.weight(1f)
                )
            }
            item.options.forEachIndexed { index, option ->
                Log.d("ItemDetail", "option: $option")
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 6.dp)
                        .background(if (index == item.selectedIndex) Color.Magenta else Color.Transparent),
                    verticalAlignment = Alignment.CenterVertically

                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 6.dp)
                            .background(if (index == selectedIndex) Color.Magenta else Color.Transparent)
                            .clickable {
                                item.selectedIndex = index
                                selectedIndex = index
                            }
                    ) {
                        Text(
                            text = "Option: $option",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onPrimary,
                            modifier = Modifier.padding(16.dp)
                        )
                    }
                }
            }
            Row {
                Button(onClick = {
                    val isAnswerCorrect = item.selectedIndex == item.indexCorrectOption
                    onAnswerSubmit(isAnswerCorrect)
                }) {
                    Text("Next")
                }
            }
        }
    }
}

//@Composable
//fun ItemDetail(
//    item: Item,
//    onAnswerSubmit: (answer : Boolean?) -> Unit
//) {
//    val gradientColors = listOf(
//        MaterialTheme.colorScheme.tertiary,
//        MaterialTheme.colorScheme.primary,
//        MaterialTheme.colorScheme.secondary
//    )
//    val textColor = MaterialTheme.colorScheme.onPrimary
//
//    Box(
//        modifier = Modifier
//            .fillMaxWidth()
//            .padding(46.dp)
//            .clip(
//                shape = RoundedCornerShape(
//                    topStart = CornerSize(5.dp),
//                    topEnd = CornerSize(30.dp),
//                    bottomStart = CornerSize(30.dp),
//                    bottomEnd = CornerSize(5.dp)
//                )
//            )
//            .background(brush = Brush.linearGradient(gradientColors))
//            .padding(12.dp)
//    ) {
//        Column {
//            Row {
//                Text(
//                    text = "Text: ${item.text}",
//                    style = MaterialTheme.typography.bodyLarge.copy(
//                        color = textColor
//                    ),
//                    modifier = Modifier.weight(1f)
//                )
//            }
//            item.options.forEachIndexed { index, option ->
//                Log.d("ItemDetail", "option: $option")
//                Row(
//                    modifier = Modifier
//                        .fillMaxWidth()
//                        .padding(bottom = 6.dp)
//                        .background(if (index == item.selectedIndex) Color.Magenta else Color.Transparent),
//                    verticalAlignment = Alignment.CenterVertically
//
//                ) {
//                    RadioButton(
//                        selected = index == item.selectedIndex,
//                        onClick = {
//                            item.selectedIndex = index
//                        },
//                        modifier = Modifier.padding(end = 8.dp)
//                    )
//                    Text(
//                        text = "Option: $option",
//                        style = MaterialTheme.typography.bodyMedium.copy(color = textColor),
//                        modifier = Modifier.weight(1f)
//                    )
//                }
//            }
//            Row {
//                Button(onClick = {
//                    val isAnswerCorrect = item.selectedIndex == item.indexCorrectOption
//                    onAnswerSubmit(isAnswerCorrect)
//                }) {
//                    Text("Next")
//                }
//            }
//        }
//    }
//}

//
//@Composable
//fun ItemDetail(id: Int, item: Item, onItemClick: OnItemFn) {
//    var isExpanded by remember { mutableStateOf(false) }
//    var isBold by remember { mutableStateOf(!item.read) }
//    Log.d("ItemDetail", "recompose for $item")
//    val itemViewModel = viewModel<ItemViewModel>(factory = ItemViewModel.Factory(id))
//    val gradientColors = listOf(
//        MaterialTheme.colorScheme.tertiary,
//        MaterialTheme.colorScheme.primary,
//        MaterialTheme.colorScheme.secondary
//    )
//    val textColor = MaterialTheme.colorScheme.onPrimary
//
//    Box(
//        modifier = Modifier
//            .fillMaxWidth()
//            .padding(6.dp)
//            .clip(
//                shape = RoundedCornerShape(
//                    topStart = CornerSize(5.dp),
//                    topEnd = CornerSize(30.dp),
//                    bottomStart = CornerSize(30.dp),
//                    bottomEnd = CornerSize(5.dp)
//                )
//            )
//            .background(brush = Brush.linearGradient(gradientColors))
//            .clickable(onClick = {
//                isExpanded = !isExpanded
//            })
//            .padding(12.dp)
//    ) {
//        Column {
//            Row(
//                modifier = Modifier.fillMaxWidth(),
//                verticalAlignment = Alignment.CenterVertically,
//                horizontalArrangement = Arrangement.SpaceBetween
//            ) {
//                Text(
//                    text = "Text: ${item.text}",
//                    style = MaterialTheme.typography.bodyLarge.copy(
//                        fontWeight = if (isBold) FontWeight.Bold else FontWeight.Normal,
//                        color = textColor
//                    ),
//                    modifier = Modifier.weight(1f)
//                )
//                Box(
//                    modifier = Modifier
//                        .clickable {
//                            onItemClick(item.id)
//                        }
//                        .padding(4.dp)
//                ) {
//                    Icon(
//                        imageVector = Icons.Default.Edit,
//                        contentDescription = "Expand/Collapse",
//                        tint = MaterialTheme.colorScheme.onPrimary
//                    )
//                }
//            }
//            Text(
//                text = "Sender: ${item.sender}",
//                style = MaterialTheme.typography.bodyMedium.copy(color = textColor),
//                modifier = Modifier.padding(bottom = 6.dp)
//            )
//            Text(
//                text = "Created: ${item.created}",
//                style = MaterialTheme.typography.bodyMedium.copy(color = textColor),
//                modifier = Modifier.padding(bottom = 6.dp)
//            )
//            Text(
//                text = "Read: ${item.read}",
//                style = MaterialTheme.typography.bodyMedium.copy(color = textColor),
//                modifier = Modifier.padding(bottom = 6.dp)
//            )
//            LaunchedEffect(Unit) {
//                delay(1000)
//                if (!item.read) {
//                    itemViewModel.saveOrUpdateItem(
//                        item.id,
//                        item.text,
//                        true,
//                        item.sender,
//                        item.created
//                    )
//                    isBold = false
//                }
//            }
//        }
//    }
//}
