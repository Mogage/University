package com.uni.exam1.core.data.remote

import com.google.gson.GsonBuilder
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object Api {
    private val url = "10.0.2.2:3000"
    private val httpUrl = "http://$url/"
    val wsUrl = "ws://$url"

    private var gson = GsonBuilder().create()

    val tokenInterceptor = TokenInterceptor()

    val okHttpClient = OkHttpClient.Builder().apply {
        this.addInterceptor(tokenInterceptor)
    }.build()

    val retrofit = Retrofit.Builder()
        .baseUrl(httpUrl)
        .addConverterFactory(GsonConverterFactory.create(gson))
        .build()
}