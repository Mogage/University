package com.uni.exam1.todo.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.uni.exam1.MyApplication
import com.uni.exam1.todo.data.ItemRepository

data class QuestionIdsUiState(
    val questionIds: List<Int> = listOf()
)

class QuestionIdsViewModel(
    private val itemRepository: ItemRepository
) : ViewModel() {

    var uiState: QuestionIdsUiState by mutableStateOf(QuestionIdsUiState())

    init {
        uiState = uiState.copy(questionIds = listOf())
    }

    fun setQuestionIds(questionIds: List<Int>) {
        uiState = uiState.copy(questionIds = questionIds)
    }

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app =
                    (this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as MyApplication)
                QuestionIdsViewModel(app.container.itemRepository)
            }
        }
    }
}