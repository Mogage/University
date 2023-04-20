package root.proiect_mpp.controllers;

import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.stage.Stage;
import root.proiect_mpp.Main;
import root.proiect_mpp.domain.people.Employee;
import root.proiect_mpp.repositories.people.employees.EmployeeRepository;
import root.proiect_mpp.service.logIn.LogInService;

import java.util.Objects;
import java.util.Properties;

public class LogInController {
    @FXML
    public TextField emailLogInInput;
    @FXML
    public PasswordField passwordLogInput;
    @FXML
    public Button logInButton;

    private LogInService employeeService;
    private Properties properties;

    public void initialise(Properties properties)
    {
        this.properties = properties;
        employeeService = new LogInService(new EmployeeRepository(properties));
    }

    private void changeScene() {
        Scene scene;
        FXMLLoader loader = new FXMLLoader(Main.class.getResource("MainView.fxml"));

        try{
            scene = new Scene(loader.load(), 800, 500);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        MainController mainController = loader.getController();
        mainController.initialise(properties);

        Stage currentStage = (Stage) emailLogInInput.getScene().getWindow();

        Stage newStage = new Stage();
        newStage.setScene(scene);
        newStage.setResizable(false);
        newStage.setTitle("Avioane");
        currentStage.close();
        newStage.show();
    }

    @FXML
    public void logInAction() {
        String email = emailLogInInput.getText();
        String password = passwordLogInput.getText();
        Alert alert;

        Employee employee = employeeService.findByEmail(email);

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

        changeScene();
    }
}
