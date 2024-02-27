
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColor
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.tween
import androidx.compose.animation.core.updateTransition
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Edit
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.unit.dp

@Composable
fun MyFloatingActionButton(
    onClick: () -> Unit,
    isEditing: Boolean
) {
    val transition = updateTransition(targetState = isEditing, label = "editTransition")
    val rotation by transition.animateFloat(label = "rotation") { state ->
        if (state) 45f else 0f
    }
    val scale by transition.animateFloat(label = "scale") { state ->
        if (state) 1.2f else 1f
    }
    val textColor by transition.animateColor(label = "textColor", transitionSpec = {
        tween(durationMillis = 500)
    }) { state ->
        if (state) Color.Blue else Color.Black
    }
    val iconColor by transition.animateColor(label = "iconColor", transitionSpec = {
        tween(durationMillis = 500)
    }) { state ->
        if (state) Color.Blue else Color.Black
    }

    FloatingActionButton(onClick = onClick) {
        Row(
            modifier = Modifier.padding(horizontal = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Default.Edit,
                contentDescription = null,
                tint = iconColor,
                modifier = Modifier
                    .graphicsLayer(
                        rotationZ = rotation,
                        scaleX = scale,
                        scaleY = scale
                    )
            )
            AnimatedVisibility(
                visible = !isEditing,
                enter = fadeIn(),
                exit = fadeOut()
            ) {
                Text(
                    text = "Add",
                    color = textColor,
                    modifier = Modifier
                        .padding(start = 8.dp, top = 3.dp)
                        .graphicsLayer(
                            rotationZ = rotation,
                            scaleX = scale,
                            scaleY = scale
                        )
                )
            }
        }
    }
}
