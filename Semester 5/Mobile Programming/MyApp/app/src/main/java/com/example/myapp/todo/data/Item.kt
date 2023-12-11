package com.example.myapp.todo.data

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.time.LocalDateTime

@Entity(tableName = "items")
data class Item(
    @PrimaryKey val _id: String = "",
    val title: String = "",
    val type: String = "",
    val noOfGuests: Int = 0,
    val startDate: String = LocalDateTime.now().toString(),
    val endDate: String = LocalDateTime.now().toString(),
    val isCompleted: Boolean = false,
    val doesRepeat: Boolean = false,
)
