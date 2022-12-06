package socialNetwork.controllers;

import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Stage;

public class LogInViewController {
    @FXML
    private Button logInButtonLogInPane;
    @FXML
    private Button signUpButtonLogInPane;
    @FXML
    private Button logInButtonRegisterPane;
    @FXML
    private Button signUpButtonRegisterPane;
    @FXML
    private AnchorPane logInPane;
    @FXML
    private AnchorPane signUpPane;


    @FXML
    public void onSignUpButtonLogInClick() {
        logInPane.setVisible(false);
        signUpButtonLogInPane.setVisible(true);
    }

    @FXML
    public void onSignUnButtonClickRegiser() {
        logInPane.setVisible(true);
        signUpButtonLogInPane.setVisible(false);
    }


}
