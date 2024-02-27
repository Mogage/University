package com.uni.exam1.todo.data

import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.TypeConverter
import androidx.room.TypeConverters
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

@Entity(tableName = "items")
data class Item(
    @PrimaryKey val _id: String = "",
    val text: String = "",
    @TypeConverters(StringListConverter::class)
    val options: List<String> = listOf(),
    val indexCorrectOption: Int = 0
)

class StringListConverter {
    @TypeConverter
    fun fromStringList(value: List<String>): String {
        return Gson().toJson(value)
    }

    @TypeConverter
    fun toStringList(value: String): List<String> {
        val listType = object : TypeToken<List<String>>() {}.type
        return Gson().fromJson(value, listType)
    }
}