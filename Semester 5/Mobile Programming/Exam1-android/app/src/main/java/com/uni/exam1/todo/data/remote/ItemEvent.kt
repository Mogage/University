package com.uni.exam1.todo.data.remote

import com.uni.exam1.todo.data.Item

data class ItemEvent(
    val type: String,
    val payload: Item
)
