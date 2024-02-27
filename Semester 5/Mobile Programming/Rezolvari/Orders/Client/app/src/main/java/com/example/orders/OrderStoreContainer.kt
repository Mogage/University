package com.example.orders


import android.content.Context
import android.util.Log
import androidx.datastore.preferences.preferencesDataStore
import com.example.orders.core.TAG
import com.example.orders.auth.data.AuthRepository
import com.example.orders.auth.data.remote.AuthDataSource
import com.example.orders.core.data.UserPreferencesRepository
import com.example.orders.core.data.remote.Api
import com.example.orders.core.utils.ConnectivityManagerNetworkMonitor
import com.example.orders.todo.data.OrderRepository
import com.example.orders.todo.data.remote.OrderService
import com.example.orders.todo.data.remote.OrderWsClient
import com.example.orders.todo.data.remote.OrderApi

val Context.userPreferencesDataStore by preferencesDataStore(
    name = "user_preferences"
)

class OrderStoreContainer(val context: Context) {
    init {
        Log.d(TAG, "init")
    }

    private val orderService: OrderService = Api.retrofit.create(OrderService::class.java)
    private val orderWsClient: OrderWsClient = OrderWsClient(Api.okHttpClient)
    private val authDataSource: AuthDataSource = AuthDataSource()

    private val database: OrderStoreAndroidDatabase by lazy { OrderStoreAndroidDatabase.getDatabase(context) }

    val orderRepository: OrderRepository by lazy {
        OrderRepository(orderService, orderWsClient, database.orderDao(), ConnectivityManagerNetworkMonitor(context))
    }

    val authRepository: AuthRepository by lazy {
        AuthRepository(authDataSource)
    }

    val userPreferencesRepository: UserPreferencesRepository by lazy {
        UserPreferencesRepository(context.userPreferencesDataStore)
    }
    // init OrderApi OrderService
    init {
        OrderApi.orderRepository = orderRepository
    }
}