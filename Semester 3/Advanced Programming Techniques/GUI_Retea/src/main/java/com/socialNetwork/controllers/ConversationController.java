package com.socialNetwork.controllers;

import com.socialNetwork.controllers.customs.MessageCellFactory;
import com.socialNetwork.domain.Friendship;
import com.socialNetwork.domain.Message;
import com.socialNetwork.domain.User;
import com.socialNetwork.domain.UserMessage;
import com.socialNetwork.exceptions.RepositoryException;
import com.socialNetwork.exceptions.ValidationException;
import com.socialNetwork.service.Service;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class ConversationController {
    public ListView<UserMessage> messagesListView;
    public ImageView sendMessageImageView;
    public TextField messageTextField;
    public Label userNameLabel;

    private Service service;
    private Friendship friendship;
    private Long loggedInUserId;
    private final ObservableList<UserMessage> messagesList = FXCollections.observableArrayList();
    private Alert alert;

    public void setService(Service service, Friendship friendship, Long loggedInUserId) {
        this.service = service;
        this.friendship = friendship;
        this.loggedInUserId = loggedInUserId;
        try {
            User friend = service.getUser(getFriendId(friendship, loggedInUserId));
            userNameLabel.setText(friend.getFirstName() + " " + friend.getLastName());
        } catch (RepositoryException e) {
            alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            alert.show();

        }
        uploadList();
    }

    private Long getFriendId(Friendship friendship, Long loggedInUserId) {
        return Objects.equals(friendship.getIdUser1(), loggedInUserId) ?
                friendship.getIdUser2() :
                friendship.getIdUser1();
    }

    private void uploadList() {
        List<UserMessage> messagesListAux = new ArrayList<>();
        List<Message> allMessages = service.getMessages(friendship.getId());
        for (Message message : allMessages) {
            UserMessage userMessage = new UserMessage(message, loggedInUserId);
            messagesListAux.add(userMessage);
        }
        messagesList.setAll(messagesListAux);
        messagesListView.setItems(messagesList);
        messagesListView.setCellFactory(new MessageCellFactory());
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
            alert.show();
        }
        messageTextField.setText("");
        uploadList();
    }

    public void onSendMessageKeyAction(KeyEvent keyEvent) {
        if (keyEvent.getCode() == KeyCode.ENTER) {
            onSendMessageAction();
        }
    }
}
