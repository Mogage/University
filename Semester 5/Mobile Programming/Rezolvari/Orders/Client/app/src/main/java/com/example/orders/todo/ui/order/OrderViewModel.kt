package com.example.orders.todo.ui.order

import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.orders.OrderStoreAndroid
import com.example.orders.core.Result
import com.example.orders.core.TAG
import com.example.orders.todo.data.Order
import com.example.orders.todo.data.OrderRepository
import kotlinx.coroutines.launch


data class OrderUiState(
    val orderId: Int? = null,
    val order: Order = Order(),
    var loadResult: Result<Order>? = null,
    var submitResult: Result<Order>? = null,
)

class OrderViewModel(private val orderId: Int?, private val orderRepository: OrderRepository) :
    ViewModel() {

    var uiState: OrderUiState by mutableStateOf(OrderUiState(loadResult = Result.Loading))
        private set

    init {
        Log.d(TAG, "init")
        if (orderId != null) {
            loadOrder()
        } else {
            uiState = uiState.copy(loadResult = Result.Success(Order()))
        }
    }

    private fun loadOrder() {
        viewModelScope.launch {
            orderRepository.orderStream.collect { orders ->
                Log.d(TAG, "Collecting orders")
                if (uiState.loadResult !is Result.Loading) {
                    return@collect
                }
                Log.d(TAG, "Collecting orders - loadResult is loading, attempting to find order with id: $orderId")
                val order = orders.find { it.code == orderId } ?: Order()
                Log.d(TAG, "Collecting orders - order: $order")
                uiState = uiState.copy(order = order, loadResult = Result.Success(order))
            }
        }
    }

    fun updateOrderWithQuantity(order: Order) {
        Log.d(TAG, "updateOrdersWithQuantity...")
        viewModelScope.launch {
            orderRepository.updateOrderWithQuantityLocally(order)
        }
    }

    fun saveOrUpdateOrder(
        id: Int,
        text: String,
        read: Boolean,
        sender: String,
        created: Long,
    ) {
        viewModelScope.launch {
            try {
                uiState = uiState.copy(submitResult = Result.Loading)

                // Convert publicationDate to a valid date format if needed
//                val formattedDate = DateUtils.parseDDMMYYYY(publicationDate)
//                if (formattedDate == null) {
//                    uiState = uiState.copy(submitResult = Result.Error(Exception("Invalid date format")))
//                    return@launch
//                }
//                val formattedDateStr = formattedDate.toString()

                val order = uiState.order.copy(
                    code = id,
                    name = text,
                )

                val savedOrder: Order = if (orderId == null) {
                    orderRepository.save(order)
                } else {
                    orderRepository.update(order)
                }

                uiState = uiState.copy(submitResult = Result.Success(savedOrder))
            } catch (e: Exception) {
                uiState = uiState.copy(submitResult = Result.Error(e))
            }
        }
    }


    companion object {
        fun Factory(orderId: Int?): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app =
                    (this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as OrderStoreAndroid)
                OrderViewModel(orderId, app.container.orderRepository)
            }
        }
    }
}
