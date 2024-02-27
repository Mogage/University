package com.example.orders

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.orders.core.TAG
import com.example.orders.core.data.UserPreferences
import com.example.orders.core.data.UserPreferencesRepository
import com.example.orders.todo.data.OrderRepository
import kotlinx.coroutines.launch

class OrderStoreViewModel (
    private val userPreferencesRepository: UserPreferencesRepository,
    private val orderRepository: OrderRepository
) :
    ViewModel() {

    init {
        Log.d(TAG, "init")
    }

    fun logout() {
        viewModelScope.launch {
            orderRepository.deleteAll()
            userPreferencesRepository.save(UserPreferences())
        }
    }

    fun setToken(token: String) {
        orderRepository.setToken(token)
    }

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app =
                    (this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as OrderStoreAndroid)
                OrderStoreViewModel(
                    app.container.userPreferencesRepository,
                    app.container.orderRepository
                )
            }
        }
    }
}
