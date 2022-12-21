package com.socialNetwork.controllers;

import com.socialNetwork.domain.Friendship;
import com.socialNetwork.domain.Message;
import com.socialNetwork.exceptions.RepositoryException;
import com.socialNetwork.exceptions.ValidationException;
import com.socialNetwork.service.Service;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.scene.control.ListView;
import javafx.scene.control.TextField;
import javafx.scene.image.ImageView;

import java.util.List;
import java.util.Objects;

public class ConversationController {
    public ListView<Message> messagesListView;
    public ImageView sendMessageImageView;
    public TextField messageTextField;

    private Service service;
    private Friendship friendship;
    private Long loggedInUserId;
    private final ObservableList<Message> messagesList = FXCollections.observableArrayList();
    private Alert alert;

    public void setService(Service service, Friendship friendship, Long loggedInUserId) {
        this.service = service;
        this.friendship = friendship;
        this.loggedInUserId = loggedInUserId;
        uploadList();
    }

    private void uploadList() {
        List<Message> messagesListAux = service.getMessages(friendship.getId());
        messagesList.setAll(messagesListAux);
        messagesListView.setItems(messagesList);
    }

    @FXML
    public void onSendMessageAction() {
        try {
            service.sendMessage(
                    friendship.getId(),
                    messageTextField.getText(),
                    loggedInUserId,
                    (Objects.equals(friendship.getIdUser1(), loggedInUserId) ?
                    friendship.getIdUser2() :
                    friendship.getIdUser1()));
        } catch (ValidationException | RepositoryException e) {
            alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
        }
        messageTextField.setText("");
        uploadList();
    }
}
