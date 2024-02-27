package com.uni.exam1

import android.content.Context
import android.util.Log
import androidx.datastore.preferences.preferencesDataStore
import com.uni.exam1.auth.data.AuthRepository
import com.uni.exam1.auth.data.remote.AuthDataSource
import com.uni.exam1.core.TAG
import com.uni.exam1.core.data.UserPreferencesRepository
import com.uni.exam1.core.data.remote.Api
import com.uni.exam1.todo.data.ItemRepository
import com.uni.exam1.todo.data.remote.ItemService
import com.uni.exam1.todo.data.remote.ItemWsClient

val Context.userPreferencesDataStore by preferencesDataStore(
    name = "user_preferences"
)

class AppContainer(val context: Context) {
    init {
        Log.d(TAG, "init")
    }

    private val itemService: ItemService = Api.retrofit.create(ItemService::class.java)
    private val itemWsClient: ItemWsClient = ItemWsClient(Api.okHttpClient)
    private val authDataSource: AuthDataSource = AuthDataSource()

    private val database: Exam1Database by lazy { Exam1Database.getDatabase(context) }

    val itemRepository: ItemRepository by lazy {
        ItemRepository(itemService, itemWsClient, database.itemDao())
    }

    val authRepository: AuthRepository by lazy {
        AuthRepository(authDataSource)
    }

    val userPreferencesRepository: UserPreferencesRepository by lazy {
        UserPreferencesRepository(context.userPreferencesDataStore)
    }
}
