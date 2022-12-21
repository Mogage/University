package com.socialNetwork.controllers;

import com.socialNetwork.MainGUI;
import com.socialNetwork.domain.DTOUserFriendship;
import com.socialNetwork.domain.Friendship;
import com.socialNetwork.domain.User;
import com.socialNetwork.exceptions.NetworkException;
import com.socialNetwork.exceptions.RepositoryException;
import com.socialNetwork.exceptions.ValidationException;
import com.socialNetwork.service.Service;
import javafx.beans.binding.Bindings;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.stage.Stage;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class MainController {
    public TableView<DTOUserFriendship> friendsTable;
    public TableView<User> searchUserTable;
    public TableView<DTOUserFriendship> requestsTable;
    public TableColumn<DTOUserFriendship, String> friendsFirstNameColumn;
    public TableColumn<DTOUserFriendship, String> friendsLastNameColumn;
    public TableColumn<DTOUserFriendship, String> friendsSinceFromColumn;
    public TableColumn<DTOUserFriendship, String> requestsFirstNameColumn;
    public TableColumn<DTOUserFriendship, String> requestsLastNameColumn;
    public TableColumn<DTOUserFriendship, String> requestsSinceFromColumn;
    public TableColumn<User, String> searchUserFirstNameColumn;
    public TableColumn<User, String> searchUserLastNameColumn;
    public Label userName;
    public TextField searchBar;
    public Button addFriendButton;
    public Button logOutButton;
    public Button deleteAccountButton;
    public Button removeFriendButton;
    public Button acceptRequestButton;
    public Button refuseRequestButton;
    public Button refreshRequestsButton;
    public Button openConversationButton;

    private Service service;
    private User loggedInUser;
    private final ObservableList<DTOUserFriendship> friendsList = FXCollections.observableArrayList();
    private final ObservableList<User> usersList = FXCollections.observableArrayList();
    private final ObservableList<DTOUserFriendship> requestsList = FXCollections.observableArrayList();
    private Alert alert;

    public void initialise(Service service, User loggedInUser) {
        this.service = service;
        this.loggedInUser = loggedInUser;

        userName.setText(loggedInUser.getFirstName() + " " + loggedInUser.getLastName());
        searchBar.textProperty().addListener(o -> onSearchUser());
        removeFriendButton.disableProperty().bind(Bindings.isEmpty(friendsTable.getSelectionModel().getSelectedItems()));
        openConversationButton.disableProperty().bind(Bindings.isEmpty(friendsTable.getSelectionModel().getSelectedItems()));
        addFriendButton.disableProperty().bind(Bindings.isEmpty(searchUserTable.getSelectionModel().getSelectedItems()));
        acceptRequestButton.disableProperty().bind(Bindings.isEmpty(requestsTable.getSelectionModel().getSelectedItems()));
        refuseRequestButton.disableProperty().bind(Bindings.isEmpty(requestsTable.getSelectionModel().getSelectedItems()));
        initTables();
    }

    private void onSearchUser() {
        List<User> userListAux = new ArrayList<>();
        Iterable<User> users = service.getAllUsers();
        String insertedText = searchBar.getText();

        for (User user : users) {
            if (user.getFirstName().contains(insertedText) || user.getLastName().contains(insertedText)) {
                userListAux.add(user);
            }
        }
        usersList.setAll(userListAux);
        searchUserTable.setItems(usersList);
    }

    private List<DTOUserFriendship> updateFriendsList(List<Friendship> friendshipList) {
        List<DTOUserFriendship> friendshipListAux = new ArrayList<>();

        for (Friendship request : friendshipList) {
            Long friendId = (Objects.equals(request.getIdUser1(), loggedInUser.getId()) ? request.getIdUser2() : request.getIdUser1());
            try {
                User user = service.getUser(friendId);
                friendshipListAux.add(new DTOUserFriendship(request.getId(), user.getFirstName(), user.getLastName(), request.getFriendsFrom()));
            } catch (RepositoryException e) {
                alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
                alert.show();
            }
        }

        return friendshipListAux;
    }

    private void updateFriendsTable() {
        List<Friendship> userFriends = service.findUserFriends(loggedInUser.getId());
        friendsList.setAll(updateFriendsList(userFriends));
        friendsTable.setItems(friendsList);
    }

    private void updateRequestsTable() {
        List<Friendship> userRequests = service.findUserRequests(loggedInUser.getId()).stream()
                .filter(friendship -> !Objects.equals(friendship.getIdUser1(), loggedInUser.getId()))
                .collect(Collectors.toList());
        requestsList.setAll(updateFriendsList(userRequests));
        requestsTable.setItems(requestsList);
    }

    private void initFriendsTable() {
        friendsFirstNameColumn.setCellValueFactory(new PropertyValueFactory<>("firstName"));
        friendsLastNameColumn.setCellValueFactory(new PropertyValueFactory<>("lastName"));
        friendsSinceFromColumn.setCellValueFactory(new PropertyValueFactory<>("friendsFrom"));
        updateFriendsTable();
    }

    private void initSearchTable() {
        List<User> userListAux = new ArrayList<>();
        Iterable<User> users = service.getAllUsers();

        for (User user : users) {
            userListAux.add(user);
        }
        usersList.setAll(userListAux);
        searchUserFirstNameColumn.setCellValueFactory(new PropertyValueFactory<>("firstName"));
        searchUserLastNameColumn.setCellValueFactory(new PropertyValueFactory<>("lastName"));
        searchUserTable.setItems(usersList);
    }

    private void initRequestsTable() {
        requestsFirstNameColumn.setCellValueFactory(new PropertyValueFactory<>("firstName"));
        requestsLastNameColumn.setCellValueFactory(new PropertyValueFactory<>("lastName"));
        requestsSinceFromColumn.setCellValueFactory(new PropertyValueFactory<>("friendsFrom"));
        updateRequestsTable();
    }

    private void initTables() {
        initFriendsTable();
        initSearchTable();
        initRequestsTable();
    }

    @FXML
    public void onAddFriendAction() {
        try {
            User userToAskFriendship = searchUserTable.getSelectionModel().getSelectedItem();
            service.makeFriends(loggedInUser.getId(), userToAskFriendship.getId());
            updateFriendsTable();
            alert = new Alert(Alert.AlertType.INFORMATION, "Friend request sent.", ButtonType.CLOSE);
        } catch (ValidationException | RepositoryException | NetworkException e) {
            alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
        }
        alert.show();
    }

    private void goToLogInStage() {
        Scene scene;
        FXMLLoader fxmlLoader = new FXMLLoader(MainGUI.class.getResource("LogIn-SignUp_view.fxml"));

        try {
            scene = new Scene(fxmlLoader.load(), 350, 500);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        LogInViewController mainController = fxmlLoader.getController();
        mainController.setService(service);

        Stage currentStage = (Stage) logOutButton.getScene().getWindow();

        Stage newStage = new Stage();
        newStage.setScene(scene);
        newStage.setResizable(false);
        newStage.setTitle("MoSocial");
        currentStage.close();
        newStage.show();
    }

    @FXML
    public void onLogOutAction() {
        goToLogInStage();
    }

    @FXML
    public void onDeleteAccountAction() {
        try {
            service.remove(loggedInUser.getId());
            alert = new Alert(Alert.AlertType.INFORMATION, "Account deleted successfully", ButtonType.CLOSE);
        } catch (RepositoryException | NetworkException e) {
            alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
        }
        alert.show();
        goToLogInStage();
    }

    @FXML
    public void onRemoveFriendAction() {
        Long friendshipToRemoveId = friendsTable.getSelectionModel().getSelectedItem().getId();
        try {
            service.removeFriends(friendshipToRemoveId);
            updateFriendsTable();
            alert = new Alert(Alert.AlertType.INFORMATION, "Friend removed successfully.", ButtonType.CLOSE);
        } catch (NetworkException | ValidationException | RepositoryException e) {
            alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
        }
        alert.show();
    }

    @FXML
    public void onOpenConversationAction() {
        Scene scene;
        FXMLLoader fxmlLoader = new FXMLLoader(MainGUI.class.getResource("ConversationView.fxml"));

        try {
            scene = new Scene(fxmlLoader.load(), 300, 450);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        service.refreshConversation();
        Long friendshipId = friendsTable.getSelectionModel().getSelectedItem().getId();
        ConversationController conversationController = fxmlLoader.getController();
        try {
            conversationController.setService(service, service.getFriendship(friendshipId), loggedInUser.getId());
        } catch (RepositoryException e) {
            alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
        }

        Stage newStage = new Stage();
        newStage.setScene(scene);
        newStage.setResizable(false);
        newStage.setTitle("MoSocial");
        newStage.show();
    }

    @FXML
    public void onAcceptRequestAction() {
        try {
            Friendship friendshipToUpdate = service.getFriendship(requestsTable.getSelectionModel().getSelectedItem().getId());
            service.updateFriends(friendshipToUpdate.getId(), friendshipToUpdate.getIdUser1(), friendshipToUpdate.getIdUser2());
            updateRequestsTable();
            updateFriendsTable();
            alert = new Alert(Alert.AlertType.INFORMATION, "Friend request accepted.", ButtonType.CLOSE);
        } catch (ValidationException | RepositoryException | NetworkException e) {
            alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
        }
        alert.show();
    }

    @FXML
    public void onRefuseRequestAction() {
        try {
            service.removeFriends(requestsTable.getSelectionModel().getSelectedItem().getId());
            updateRequestsTable();
            alert = new Alert(Alert.AlertType.CONFIRMATION, "Friend request refused.", ButtonType.CLOSE);
        } catch (NetworkException | ValidationException | RepositoryException e) {
            alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
        }
        alert.show();
    }

    @FXML
    public void onRefreshRequestsAction() {
        service.refresh();
        updateFriendsTable();
        updateRequestsTable();
    }
}
