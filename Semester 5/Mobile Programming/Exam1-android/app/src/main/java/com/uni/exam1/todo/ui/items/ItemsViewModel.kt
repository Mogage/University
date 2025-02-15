package com.uni.exam1.todo.ui.items

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.uni.exam1.MyApplication
import com.uni.exam1.core.TAG
import com.uni.exam1.todo.data.Item
import com.uni.exam1.todo.data.ItemRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch

class ItemsViewModel(private val itemRepository: ItemRepository) : ViewModel() {
    val uiState: Flow<List<Item>> = itemRepository.itemStream

    init {
        Log.d(TAG, "init")
        loadItems()
    }

    fun loadItems() {
        Log.d(TAG, "loadItems...")
        viewModelScope.launch {
            itemRepository.refresh()
        }
    }

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app =
                    (this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as MyApplication)
                ItemsViewModel(app.container.itemRepository)
            }
        }
    }
}
