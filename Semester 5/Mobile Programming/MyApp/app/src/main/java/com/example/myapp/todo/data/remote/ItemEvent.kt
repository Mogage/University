package com.example.myapp.todo.data.remote

import com.example.myapp.todo.data.Item

data class ItemEvent(val type: String, val payload: Item)
