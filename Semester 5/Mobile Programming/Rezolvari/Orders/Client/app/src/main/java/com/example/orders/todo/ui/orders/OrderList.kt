package com.example.orders.todo.ui.orders


import android.util.Log
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutLinearInEasing
import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CornerSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.core.text.isDigitsOnly
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.orders.todo.data.Order
import com.example.orders.todo.ui.order.OrderViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

typealias OnOrderFn = (id: Int?) -> Unit

enum class FilterOption {
    SHOW_ALL,
    SHOW_WITH_QUANTITIES
}

@Composable
fun OrderList(
    orderList: List<Order>,
    onOrderClick: OnOrderFn,
    modifier: Modifier,
    ordersViewModel: OrdersViewModel
) {
    Log.d("OrderList", "recompose")

    var filterOption by remember { mutableStateOf(FilterOption.SHOW_ALL) }

    // Radio button for filtering options

    val filteredOrders = when (filterOption) {
        FilterOption.SHOW_ALL -> orderList
        FilterOption.SHOW_WITH_QUANTITIES -> orderList.filter { it.quantity != null && it.quantity > 0 }
    }

    var failedOrders by remember { mutableStateOf<List<Order>>(emptyList()) }
    var isError by remember { mutableStateOf(false) }
    var text by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(true) }

    Box(
        modifier = modifier
            .fillMaxSize()
            .padding(12.dp)
    ) {

        if (isLoading) {
            CircularProgressIndicator(
                modifier = Modifier
                    .height(4.dp)
                    .align(Alignment.TopCenter)
            )
        }

        Column {
            Row(
                modifier = Modifier
                    .padding(start = 8.dp)
            ) {
                RadioButton(
                    selected = filterOption == FilterOption.SHOW_ALL,
                    onClick = { filterOption = FilterOption.SHOW_ALL },
                    modifier = Modifier.padding(end = 4.dp)
                )
                Text("All")

                RadioButton(
                    selected = filterOption == FilterOption.SHOW_WITH_QUANTITIES,
                    onClick = { filterOption = FilterOption.SHOW_WITH_QUANTITIES },
                    modifier = Modifier.padding(start = 16.dp, end = 4.dp)
                )
                Text("Quantities")
            }

            LazyColumn(
                modifier = Modifier
                    .weight(0.9f)
            ) {
                items(filteredOrders) {
                    val failed = failedOrders.find { f -> f == it }
                    val isFailed = failed != null// || (it.quantity != null && it.quantity < 0)
                    OrderDetail(id = it.code, order = it, isHighlighted = isFailed)
                }
            }
            LaunchedEffect(Unit) {
                isLoading = false
            }

            val coroutineScope = rememberCoroutineScope()
            Row {

                Button(
                    onClick = {
                        // Reset failed orders
                        failedOrders = emptyList()

                        coroutineScope.launch {
                            isLoading = true
                            for (order in orderList) {
                                if (order.quantity != null) {
                                    // Call the updateOrderWithQuantity function and handle highlighting
                                    val result =
                                        ordersViewModel.updateOrderWithQuantity(order) { isSuccess ->
                                            if (!isSuccess) {
                                                failedOrders = failedOrders + order
                                            }
                                        }
                                    Log.d("OrderList", "result = $result")
                                    if (result != "OK") {
                                        isError = true
                                        text = result
                                        failedOrders = failedOrders + order
                                    }
                                }
                            }
                            isLoading = false
                        }
                    },
                    modifier = Modifier.align(Alignment.Bottom)
                ) {
                    Text(text = "Submit")
                }
            }

        }

        LaunchedEffect(isError) {
            if (!isError) {
                delay(3500L)
                text = ""
            } else {
                delay(5000L)
                isError = false
            }
        }
        ErrorInOrder(error = isError, text = text)
    }
}

@Composable
fun ErrorInOrder(error: Boolean, text: String) {
    Log.d("CanAddAOrder", "canSave = $error")
    AnimatedVisibility(
        visible = error && text.isNotEmpty(),
        enter = slideInVertically(
            initialOffsetY = { fullHeight -> -fullHeight },
            animationSpec = tween(durationMillis = 1500, easing = LinearOutSlowInEasing)
        ),
        exit = slideOutVertically(
            targetOffsetY = { fullHeight -> -fullHeight },
            animationSpec = tween(durationMillis = 1500, easing = FastOutLinearInEasing)
        )
    ) {
        Surface(
            tonalElevation = 100.dp,
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 15.dp, vertical = 60.dp)
                .clip(RoundedCornerShape(30.dp))
                .background(
                    brush = Brush.horizontalGradient(
                        listOf(
                            Color.Red,
                            MaterialTheme.colorScheme.tertiary,
                            MaterialTheme.colorScheme.primary,
                            Color.Red
                        )
                    )
                )

        ) {
            Text(
                text = text,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(15.dp),
                color = MaterialTheme.colorScheme.primary
            )
        }
    }
}


@Composable
fun OrderDetail(id: Int, order: Order, isHighlighted: Boolean = false) {
    Log.d("OrderDetail", "recompose for $order with $isHighlighted")
    val orderViewModel = viewModel<OrderViewModel>(factory = OrderViewModel.Factory(id))
    val textColor = Color.Black
    var isEditingQuantity by remember { mutableStateOf(true) }
    var isError by remember { mutableStateOf(false) }
    var text by remember { mutableStateOf("") }
    var showButton by remember { mutableStateOf(false) }
    var quantityByRemember by remember { mutableStateOf("") }

    LaunchedEffect(order.quantity){
        if (quantityByRemember != order.quantity.toString())
        {
            quantityByRemember = if (order.quantity.toString() == "null") "0" else order.quantity.toString()
        }
    }
    LaunchedEffect(isError) {
        if (!isError) {
            delay(3500L)
            text = ""
        } else {
            delay(5000L)
            isError = false
        }
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(6.dp)
            .padding(12.dp)
            .clickable { isEditingQuantity = true },
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = "Text: ${order.name}",
            style = MaterialTheme.typography.bodyLarge.copy(
                color = if (isHighlighted) Color.Red else textColor // Highlight based on the parameter
            ),
            modifier = Modifier.weight(1f)
        )
        OutlinedTextField(
            value = quantityByRemember,
            onValueChange = {
                showButton = true
                quantityByRemember = it
            },
            readOnly = !isEditingQuantity,
            textStyle = MaterialTheme.typography.bodyLarge.copy(
                color = if (isHighlighted) Color.Red else textColor
            ),
            modifier = Modifier.width(90.dp).clickable { isEditingQuantity = true },
        )
        if (showButton) {
            Button(onClick = {
                try {
                    val q = quantityByRemember.toInt()
                    val newOrder = order.copy(quantity = q)
                    orderViewModel.updateOrderWithQuantity(newOrder)
                } catch (e: Exception) {
                    Log.d("OrderDetail", "Error updating order with quantity: ${e.message}")
                    isError = true
                    text = e.message ?: "Server indisponibil"
                }
                isEditingQuantity = false
                showButton = false
            }) {
                Text(text = "Confirm")
            }
        }
    }
}
