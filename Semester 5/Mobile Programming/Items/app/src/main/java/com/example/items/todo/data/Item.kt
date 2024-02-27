package com.example.items.todo.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "items")
data class Item(
    @PrimaryKey
    val id: Int = 0,
    val text: String = "",
    val options: List<Int> = listOf(),
    val indexCorrectOption: Int = 0,
    val dirty: Boolean? = false,
    var selectedIndex: Int? = null
) {
    override fun toString(): String {
        return "Item(id=$id, text='$text', options=$options, indexCorrectOption=$indexCorrectOption, dirty=$dirty, selectedIndex=$selectedIndex)"
    }
}