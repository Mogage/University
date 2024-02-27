package com.uni.exam1.todo.data.remote

import com.uni.exam1.todo.data.Item
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Header
import retrofit2.http.Headers
import retrofit2.http.POST
import retrofit2.http.PUT
import retrofit2.http.Path

interface ItemService {
    @GET("/question/0")
    suspend fun find
                (@Header("Authorization") authorization: String
    ): Item

    @GET("/question/{id}")
    suspend fun read(
        @Header("Authorization") authorization: String,
        @Path("id") itemId: String?
    ): Item;

    @Headers("Content-Type: application/json")
    @POST("/question/0")
    suspend fun create(@Header("Authorization") authorization: String, @Body item: Item): Item

    @Headers("Content-Type: application/json")
    @PUT("/question/{id}")
    suspend fun update(
        @Header("Authorization") authorization: String,
        @Path("id") itemId: String?,
        @Body item: Item
    ): Item

    @Headers("Content-Type: application/json")
    @PUT("/question/sync")
    suspend fun sync(
        @Header("Authorization") authorization: String,
        @Body items: List<Item>
    ): List<Item>
}
