package com.socialNetwork.controllers;

import com.socialNetwork.MainGUI;
import com.socialNetwork.domain.User;
import com.socialNetwork.exceptions.RepositoryException;
import com.socialNetwork.exceptions.ValidationException;
import com.socialNetwork.service.Service;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Stage;

import java.util.Objects;

public class LogInViewController {
    @FXML
    public Button logInButtonLogInPane;
    @FXML
    public Button signUpButtonLogInPane;
    @FXML
    public Label logInLabelRegisterPane;
    @FXML
    public Button signUpButtonRegisterPane;
    @FXML
    public AnchorPane logInPane;
    @FXML
    public AnchorPane signUpPane;
    @FXML
    public TextField firstNameRegisterTextField;
    @FXML
    public TextField lastNameRegisterTextField;
    @FXML
    public TextField emailRegisterTextField;
    @FXML
    public TextField emailLogInTextField;
    @FXML
    public PasswordField passwordRegisterTextField;
    @FXML
    public PasswordField repeatPasswordRegisterTextField;
    @FXML
    public PasswordField passwordLogInTextField;
    private Service service;

    public void setService(Service service) {
        this.service = service;
    }

    @FXML
    public void onSignUpButtonLogInClick() {
        logInPane.setVisible(false);
        signUpPane.setVisible(true);
    }

    @FXML
    public void onSignUpButtonRegisterClick() {
        String firstName = firstNameRegisterTextField.getText();
        String lastName = lastNameRegisterTextField.getText();
        String email = emailRegisterTextField.getText();
        String password = passwordRegisterTextField.getText();
        String repeatPassword = repeatPasswordRegisterTextField.getText();
        if (!Objects.equals(password, repeatPassword)) {
            Alert alert = new Alert(Alert.AlertType.ERROR, "Passwords are different!", ButtonType.OK);
            alert.show();
            return;
        }
        System.out.println(firstName + lastName + email + password);

        try {
            service.add(firstName, lastName, email, password);
        } catch (ValidationException | RepositoryException e) {
            Alert alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            alert.show();
        }
    }

    @FXML
    public void onLogInButtonRegisterClick() {
        logInPane.setVisible(true);
        signUpPane.setVisible(false);
    }

    private void changeScene() {
        Scene scene;
        FXMLLoader loader = new FXMLLoader(MainGUI.class.getResource("MainView.fxml"));

        try{
            scene = new Scene(loader.load(), 801, 501);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        Stage currentStage = (Stage) logInButtonLogInPane.getScene().getWindow();

        Stage newStage = new Stage();
        newStage.setScene(scene);
        newStage.setResizable(false);
        newStage.setTitle("MoSocial");
        currentStage.close();
        newStage.show();
    }

    @FXML
    public void onLogInButtonLogInClick() {
        String email = emailLogInTextField.getText();
        String password = passwordLogInTextField.getText();
        try {
            User user = service.findUserAfterEmail(email);
            if (!Objects.equals(password, user.getPassword())) {
                Alert alert = new Alert(Alert.AlertType.ERROR, "Password does not match", ButtonType.OK);
                alert.show();
            }
            changeScene();
        } catch (RepositoryException e) {
            Alert alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            alert.show();
        }
    }

}
