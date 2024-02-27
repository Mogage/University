package com.uni.exam1

import android.content.Context
import androidx.room.Database
import androidx.room.Room.databaseBuilder
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import com.uni.exam1.todo.data.Item
import com.uni.exam1.todo.data.StringListConverter
import com.uni.exam1.todo.data.local.ItemDao

@Database(entities = [Item::class], version = 1)
@TypeConverters(StringListConverter::class)
abstract class Exam1Database : RoomDatabase() {
    abstract fun itemDao(): ItemDao

    companion object {
        @Volatile
        private var INSTANCE: Exam1Database? = null

        fun getDatabase(context: Context): Exam1Database {
            return INSTANCE ?: synchronized(this) {
                val instance = databaseBuilder(
                    context,
                    Exam1Database::class.java,
                    "myapp_database_v1"
                )
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
