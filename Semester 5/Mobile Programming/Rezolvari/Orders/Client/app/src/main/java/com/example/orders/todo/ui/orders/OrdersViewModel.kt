package com.example.orders.todo.ui.orders

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.orders.OrderStoreAndroid
import com.example.orders.core.TAG
import com.example.orders.todo.data.Order
import com.example.orders.todo.data.OrderRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

class OrdersViewModel(private val orderRepository: OrderRepository) : ViewModel() {
    val uiState: Flow<List<Order>> = orderRepository.orderStream

    init {
        Log.d(TAG, "init")
        // loadOrders()
    }

    private fun loadOrders() {
        Log.d(TAG, "loadOrders...")
        viewModelScope.launch {
            orderRepository.refresh()
        }
    }

    suspend fun updateOrderWithQuantity(order: Order, onResult: (Boolean) -> Unit): String {
        Log.d(TAG, "updateOrdersWithQuantity...")
        var result = ""
        suspendCoroutine { continuation ->
            viewModelScope.launch {
                result = orderRepository.updateOrderWithQuantity(order) { isSuccess ->
                    onResult(isSuccess)
                    continuation.resume(if (isSuccess) "Success" else "Failure")
                }
                Log.d(TAG, "updateOrdersWithQuantity result: $result")
            }
        }
        return result
    }


    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app =
                    (this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as OrderStoreAndroid)
                OrdersViewModel(app.container.orderRepository)
            }
        }
    }
}
