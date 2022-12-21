package com.socialNetwork.controllers.customs;

import com.socialNetwork.domain.UserMessage;
import javafx.geometry.Pos;
import javafx.scene.control.ListCell;
import javafx.scene.control.ListView;
import javafx.util.Callback;

import java.util.Objects;

public class MessageCellFactory implements Callback<ListView<UserMessage>, ListCell<UserMessage>> {
    @Override
    public ListCell<UserMessage> call(ListView<UserMessage> param) {
        return new ListCell<>() {
            @Override
            public void updateItem(UserMessage message, boolean empty) {
                super.updateItem(message, empty);
                if (empty || message == null) {
                    setText(null);
                } else {
                    setText(message.getText());
                    if (Objects.equals(message.getCurrentUserId(), message.getFromUserId())) {
                        setAlignment(Pos.CENTER_RIGHT);
                    } else {
                        setAlignment(Pos.CENTER_LEFT);
                    }
                }
            }
        };
    }
}

