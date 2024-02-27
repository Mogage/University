package com.uni.exam1

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.uni.exam1.core.TAG
import com.uni.exam1.core.data.UserPreferences
import com.uni.exam1.core.data.UserPreferencesRepository
import com.uni.exam1.todo.data.ItemRepository
import kotlinx.coroutines.launch

class Exam1ViewModel (
    private val userPreferencesRepository: UserPreferencesRepository,
    private val itemRepository: ItemRepository
) : ViewModel() {

    init {
        Log.d(TAG, "init")
    }

    fun logout() {
        viewModelScope.launch {
            itemRepository.deleteAll()
            userPreferencesRepository.save(UserPreferences())
        }
    }

    fun setToken(token: String) {
        itemRepository.setToken(token)
    }

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app =
                    (this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as MyApplication)
                Exam1ViewModel(
                    app.container.userPreferencesRepository,
                    app.container.itemRepository
                )
            }
        }
    }
}
