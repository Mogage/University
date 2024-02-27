package com.uni.exam1.auth

import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.uni.exam1.MyApplication
import com.uni.exam1.auth.data.AuthRepository
import com.uni.exam1.core.data.UserPreferences
import com.uni.exam1.core.data.UserPreferencesRepository
import com.uni.exam1.todo.data.ItemRepository
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

data class LoginUiState(
    val isAuthenticating: Boolean = false,
    val authenticationError: Throwable? = null,
    val authenticationCompleted: Boolean = false,
    val isDownloading: Boolean = false,
    val downloadError: Throwable? = null,
    val downloadCompleted: Boolean = false,
    val token: String = "",
    val toDownload: Int = 0,
    val downloaded: Int = 0,
)

class LoginViewModel(
    private val authRepository: AuthRepository,
    private val itemRepository: ItemRepository,
    private val userPreferencesRepository: UserPreferencesRepository
) : ViewModel() {

    var uiState: LoginUiState by mutableStateOf(LoginUiState())

    init {
        Log.d(TAG, "init")
    }

    fun login(username: String) {
        viewModelScope.launch {
            Log.v(TAG, "login...");
            uiState = uiState.copy(isAuthenticating = true, authenticationError = null)
            val result = authRepository.login(username)
            if (result.isSuccess) {
                userPreferencesRepository.save(
                    UserPreferences(
                        username,
                        result.getOrNull()?.token ?: ""
                    )
                )
                val questionIds = result.getOrNull()?.questionIds ?: listOf()
                uiState = uiState.copy(
                    isAuthenticating = false,
                    authenticationCompleted = true,
                    isDownloading = true,
                    toDownload = questionIds.size
                )
                questionIds.forEach { questionId ->
                    val resultDownload = itemRepository.downloadItem(questionId.toString())
                    uiState = if (resultDownload.isSuccess) {
                        uiState.copy(downloaded = uiState.downloaded + 1)
                    } else {
                        uiState.copy(
                            isDownloading = false,
                            downloadError = resultDownload.exceptionOrNull()
                        )
                    }
                }
                uiState = uiState.copy(
                    isDownloading = false,
                    downloadCompleted = true,
                )
            } else {
                uiState = uiState.copy(
                    isAuthenticating = false,
                    authenticationError = result.exceptionOrNull()
                )
            }
        }
    }

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app =
                    (this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as MyApplication)
                LoginViewModel(
                    app.container.authRepository,
                    app.container.itemRepository,
                    app.container.userPreferencesRepository
                )
            }
        }
    }
}