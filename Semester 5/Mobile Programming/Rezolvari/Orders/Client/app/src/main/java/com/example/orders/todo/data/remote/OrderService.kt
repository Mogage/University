package com.example.orders.todo.data.remote

import com.example.orders.todo.data.Item
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Header
import retrofit2.http.Headers
import retrofit2.http.POST
import retrofit2.http.PUT
import retrofit2.http.Path
import retrofit2.Response
import com.example.orders.todo.data.Order
import com.google.gson.annotations.SerializedName

interface OrderService {
    @GET("/item")
    suspend fun find(): List<Order>

    @GET("/order")
    suspend fun findWithAuth(@Header("Authorization") authorization: String): List<Order>

    @GET("/order/{id}")
    suspend fun read(
        @Header("Authorization") authorization: String,
        @Path("id") orderId: Int?
    ): Order;

    @Headers("Content-Type: application/json")
    @POST("/item")
    suspend fun create(@Body order: Order): Order

    @Headers("Content-Type: application/json")
    @POST("/order")
    suspend fun createWithAuth(@Header("Authorization") authorization: String, @Body order: Order): Order

    @Headers("Content-Type: application/json")
    @PUT("/item/{id}")
    suspend fun update(
        @Path("id") orderId: Int?,
        @Body order: Order
    ): Order

    @Headers("Content-Type: application/json")
    @POST("/item")
    suspend fun postItem(
        @Body item: Item
    ): Response<ResponseModel>

    @Headers("Content-Type: application/json")
    @PUT("/order/{id}")
    suspend fun updateWithAuth(
        @Header("Authorization") authorization: String,
        @Path("id") orderId: Int?,
        @Body order: Order
    ): Order

}

data class ResponseModel(
    val text: String = "",
    val id: Int = 0,
    val code: Int = 0,
    val quantity: Int = 0,
)
