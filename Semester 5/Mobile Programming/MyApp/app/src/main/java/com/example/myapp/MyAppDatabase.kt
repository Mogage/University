package com.example.myapp

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.Room.databaseBuilder
import androidx.room.RoomDatabase
import com.example.myapp.todo.data.Item
import com.example.myapp.todo.data.local.ItemDao


@Database(entities = arrayOf(Item::class), version = 1)
abstract class MyAppDatabase : RoomDatabase() {
    abstract fun itemDao(): ItemDao


    companion object {
        @Volatile
        private var INSTANCE: MyAppDatabase? = null

        fun getDatabase(context: Context): MyAppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = databaseBuilder(
                    context,
                    MyAppDatabase::class.java,
                    "myapp_database_v3"
                )
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
