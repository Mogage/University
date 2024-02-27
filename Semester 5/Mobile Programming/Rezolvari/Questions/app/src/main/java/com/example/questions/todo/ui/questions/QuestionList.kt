package com.example.questions.todo.ui.questions


import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.CornerSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.questions.todo.data.Question
import kotlinx.coroutines.delay

typealias OnQuestionFn = (id: Int?) -> Unit

@Composable
fun QuestionList(questionList: List<Question>, onQuestionClick: OnQuestionFn, modifier: Modifier) {
    Log.d("QuestionList", "recompose")

    var currentQuestionIndex by remember { mutableIntStateOf(0) }
    var correctAnswers by remember { mutableIntStateOf(0) }
    var totalAnsweredQuestions by remember { mutableIntStateOf(0) }
    var awaitingSubmit by remember { mutableStateOf(true) }

    QuestionHeader(
        totalAnsweredQuestions = totalAnsweredQuestions,
        totalQuestions = questionList.size,
        correctAnswers = correctAnswers
    )

    if (currentQuestionIndex < questionList.size) {
        // Show the current question
        Column(
            modifier = Modifier.fillMaxSize()
        ) {
            // Display the header with information about answered questions and correct answers

            Text(text = "\n\n")
            // Show the current question
            QuestionDetail(questionList[currentQuestionIndex], onAnswerSubmit =
            {
                    answer ->
                run {
                    Log.d("QuestionList", "Answer submitted is correct: $answer")
                    totalAnsweredQuestions++
                    if (answer != false) {
                        correctAnswers++
                    }
                    // Set the flag to indicate that submit was pressed
                    awaitingSubmit = false
                    // Increment currentQuestionIndex after the answer is submitted
                    currentQuestionIndex++
                }
            })
        }

        // Use a LaunchedEffect to automatically move to the next question after 10 seconds
        LaunchedEffect(currentQuestionIndex, awaitingSubmit) {
            Log.d("QuestionList", "LaunchedEffect start waiting for 10 seconds")
            delay(10000)
            // Check if submit was not pressed during the 10 seconds
            if (awaitingSubmit) {
                currentQuestionIndex++
                totalAnsweredQuestions++
            }
            else
            {
                awaitingSubmit = true
            }
            Log.d("QuestionList", "LaunchedEffect end waiting for 10 seconds")
        }
    } else {
        // All questions answered
        Text("All questions answered")
    }
}


@Composable
fun QuestionHeader(
    totalAnsweredQuestions: Int,
    totalQuestions: Int,
    correctAnswers: Int
) {
    // Display the header with information about answered questions and correct answers
    Text(
        text = "\n\n\n\t\tQuestions $totalAnsweredQuestions/$totalQuestions, Correct answers: $correctAnswers/$totalAnsweredQuestions",
        style = MaterialTheme.typography.bodyMedium
    )
}

@Composable
fun QuestionDetail(
    question: Question,
    onAnswerSubmit: (answer : Boolean?) -> Unit
) {
    val gradientColors = listOf(
        MaterialTheme.colorScheme.tertiary,
        MaterialTheme.colorScheme.primary,
        MaterialTheme.colorScheme.secondary
    )
    val textColor = MaterialTheme.colorScheme.onPrimary
    var selectedIndex by remember { mutableStateOf(question.selectedIndex) }
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(46.dp)
            .clip(
                shape = RoundedCornerShape(
                    topStart = CornerSize(5.dp),
                    topEnd = CornerSize(30.dp),
                    bottomStart = CornerSize(30.dp),
                    bottomEnd = CornerSize(5.dp)
                )
            )
            .background(brush = Brush.linearGradient(gradientColors))
            .padding(12.dp)
    ) {
        Column {
            Row {
                Text(
                    text = "Text: ${question.text}",
                    style = MaterialTheme.typography.bodyLarge.copy(
                        color = textColor
                    ),
                    modifier = Modifier.weight(1f)
                )
            }
            question.options.forEachIndexed { index, option ->
                Log.d("QuestionDetail", "option: $option")
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 6.dp)
                        .background(if (index == question.selectedIndex) Color.Magenta else Color.Transparent),
                    verticalAlignment = Alignment.CenterVertically

                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 6.dp)
                            .background(if (index == selectedIndex) Color.Magenta else Color.Transparent)
                            .clickable {
                                question.selectedIndex = index
                                selectedIndex = index
                            }
                    ) {
                        Text(
                            text = "Option: $option",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onPrimary,
                            modifier = Modifier.padding(16.dp)
                        )
                    }
                }
            }
            Row {
                Button(onClick = {
                    val isAnswerCorrect = question.selectedIndex == question.indexCorrectOption
                    onAnswerSubmit(isAnswerCorrect)
                }) {
                    Text("Next")
                }
            }
        }
    }
}

//@Composable
//fun QuestionDetail(
//    question: Question,
//    onAnswerSubmit: (answer : Boolean?) -> Unit
//) {
//    val gradientColors = listOf(
//        MaterialTheme.colorScheme.tertiary,
//        MaterialTheme.colorScheme.primary,
//        MaterialTheme.colorScheme.secondary
//    )
//    val textColor = MaterialTheme.colorScheme.onPrimary
//
//    Box(
//        modifier = Modifier
//            .fillMaxWidth()
//            .padding(46.dp)
//            .clip(
//                shape = RoundedCornerShape(
//                    topStart = CornerSize(5.dp),
//                    topEnd = CornerSize(30.dp),
//                    bottomStart = CornerSize(30.dp),
//                    bottomEnd = CornerSize(5.dp)
//                )
//            )
//            .background(brush = Brush.linearGradient(gradientColors))
//            .padding(12.dp)
//    ) {
//        Column {
//            Row {
//                Text(
//                    text = "Text: ${question.text}",
//                    style = MaterialTheme.typography.bodyLarge.copy(
//                        color = textColor
//                    ),
//                    modifier = Modifier.weight(1f)
//                )
//            }
//            question.options.forEachIndexed { index, option ->
//                Log.d("QuestionDetail", "option: $option")
//                Row(
//                    modifier = Modifier
//                        .fillMaxWidth()
//                        .padding(bottom = 6.dp)
//                        .background(if (index == question.selectedIndex) Color.Magenta else Color.Transparent),
//                    verticalAlignment = Alignment.CenterVertically
//
//                ) {
//                    RadioButton(
//                        selected = index == question.selectedIndex,
//                        onClick = {
//                            question.selectedIndex = index
//                        },
//                        modifier = Modifier.padding(end = 8.dp)
//                    )
//                    Text(
//                        text = "Option: $option",
//                        style = MaterialTheme.typography.bodyMedium.copy(color = textColor),
//                        modifier = Modifier.weight(1f)
//                    )
//                }
//            }
//            Row {
//                Button(onClick = {
//                    val isAnswerCorrect = question.selectedIndex == question.indexCorrectOption
//                    onAnswerSubmit(isAnswerCorrect)
//                }) {
//                    Text("Next")
//                }
//            }
//        }
//    }
//}

//
//@Composable
//fun QuestionDetail(id: Int, question: Question, onQuestionClick: OnQuestionFn) {
//    var isExpanded by remember { mutableStateOf(false) }
//    var isBold by remember { mutableStateOf(!question.read) }
//    Log.d("QuestionDetail", "recompose for $question")
//    val questionViewModel = viewModel<QuestionViewModel>(factory = QuestionViewModel.Factory(id))
//    val gradientColors = listOf(
//        MaterialTheme.colorScheme.tertiary,
//        MaterialTheme.colorScheme.primary,
//        MaterialTheme.colorScheme.secondary
//    )
//    val textColor = MaterialTheme.colorScheme.onPrimary
//
//    Box(
//        modifier = Modifier
//            .fillMaxWidth()
//            .padding(6.dp)
//            .clip(
//                shape = RoundedCornerShape(
//                    topStart = CornerSize(5.dp),
//                    topEnd = CornerSize(30.dp),
//                    bottomStart = CornerSize(30.dp),
//                    bottomEnd = CornerSize(5.dp)
//                )
//            )
//            .background(brush = Brush.linearGradient(gradientColors))
//            .clickable(onClick = {
//                isExpanded = !isExpanded
//            })
//            .padding(12.dp)
//    ) {
//        Column {
//            Row(
//                modifier = Modifier.fillMaxWidth(),
//                verticalAlignment = Alignment.CenterVertically,
//                horizontalArrangement = Arrangement.SpaceBetween
//            ) {
//                Text(
//                    text = "Text: ${question.text}",
//                    style = MaterialTheme.typography.bodyLarge.copy(
//                        fontWeight = if (isBold) FontWeight.Bold else FontWeight.Normal,
//                        color = textColor
//                    ),
//                    modifier = Modifier.weight(1f)
//                )
//                Box(
//                    modifier = Modifier
//                        .clickable {
//                            onQuestionClick(question.id)
//                        }
//                        .padding(4.dp)
//                ) {
//                    Icon(
//                        imageVector = Icons.Default.Edit,
//                        contentDescription = "Expand/Collapse",
//                        tint = MaterialTheme.colorScheme.onPrimary
//                    )
//                }
//            }
//            Text(
//                text = "Sender: ${question.sender}",
//                style = MaterialTheme.typography.bodyMedium.copy(color = textColor),
//                modifier = Modifier.padding(bottom = 6.dp)
//            )
//            Text(
//                text = "Created: ${question.created}",
//                style = MaterialTheme.typography.bodyMedium.copy(color = textColor),
//                modifier = Modifier.padding(bottom = 6.dp)
//            )
//            Text(
//                text = "Read: ${question.read}",
//                style = MaterialTheme.typography.bodyMedium.copy(color = textColor),
//                modifier = Modifier.padding(bottom = 6.dp)
//            )
//            LaunchedEffect(Unit) {
//                delay(1000)
//                if (!question.read) {
//                    questionViewModel.saveOrUpdateQuestion(
//                        question.id,
//                        question.text,
//                        true,
//                        question.sender,
//                        question.created
//                    )
//                    isBold = false
//                }
//            }
//        }
//    }
//}
