package com.uni.exam1.auth.data

import android.util.Log
import com.uni.exam1.auth.data.remote.AuthDataSource
import com.uni.exam1.auth.data.remote.TokenHolder
import com.uni.exam1.auth.data.remote.User
import com.uni.exam1.core.TAG
import com.uni.exam1.core.data.remote.Api

class AuthRepository(private val authDataSource: AuthDataSource) {
    init {
        Log.d(TAG, "init")
    }

    fun clearToken() {
        Api.tokenInterceptor.token = null
    }

    suspend fun login(username: String): Result<TokenHolder> {
        val user = User(username)
        val result = authDataSource.login(user)
        if (result.isSuccess) {
            Api.tokenInterceptor.token = result.getOrNull()?.token
        }
        return result
    }
}
