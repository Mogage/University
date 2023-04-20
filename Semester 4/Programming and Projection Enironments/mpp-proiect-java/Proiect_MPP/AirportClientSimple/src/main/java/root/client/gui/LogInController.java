package root.client.gui;

import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import root.client.StartRpcClient;
import root.model.people.Employee;
import root.services.IService;

import java.util.Objects;

public class LogInController {
    @FXML
    public TextField emailLogInInput;
    @FXML
    public PasswordField passwordLogInput;
    @FXML
    public Button logInButton;

    private IService service;
    private Parent root;
    MainController mainController;
    private final Stage stage = new Stage();

    public void setServer(IService service) {
        this.service = service;
    }

    public void setParent(Parent root) {
        this.root = root;
    }

    public void setMainController(MainController mainController) {
        this.mainController = mainController;
    }

    @FXML
    public void logInAction() {
        String email = emailLogInInput.getText();
        String password = passwordLogInput.getText();
        Alert alert;


        try {
            Employee employeeToLogIn = new Employee(email, password);
            Employee employee = service.login(employeeToLogIn, mainController);
            if (null == employee) {
                alert = new Alert(Alert.AlertType.ERROR, "Wrong email", ButtonType.OK);
                alert.show();
                return;
            }
            if (!Objects.equals(employee.getPassword(), password)) {
                alert = new Alert(Alert.AlertType.ERROR, "Wrong password", ButtonType.OK);
                alert.show();
                return;
            }
            mainController.setServer(service);
            mainController.setLoggedUser(employee);
            mainController.setStage((Stage) logInButton.getScene().getWindow());
            mainController.initialise();

            stage.setTitle("Airline");
            if (stage.getScene() == null)
                stage.setScene(new Scene(root, 800, 500));
            stage.setOnCloseRequest(event -> mainController.logOutAction());

            stage.show();
            Stage currentStage = (Stage) logInButton.getScene().getWindow();
            currentStage.close();
        } catch (Exception e) {
            alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            alert.show();
        }

        //changeScene();
    }
}
