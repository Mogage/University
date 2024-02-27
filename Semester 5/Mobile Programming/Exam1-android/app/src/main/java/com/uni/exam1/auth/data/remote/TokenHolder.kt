package com.uni.exam1.auth.data.remote

data class TokenHolder(
    var token: String,
    var questionIds: List<Int>
)