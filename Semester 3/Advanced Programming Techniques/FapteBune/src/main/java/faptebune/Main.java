package faptebune;

import faptebune.controllers.LogInSignUpController;
import faptebune.repository.NevoiRepository;
import faptebune.repository.PersoaneRepository;
import faptebune.service.Service;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

public class Main extends Application {
    String url = "jdbc:postgresql://localhost:5432/clinica";
    String userName = "postgres";
    String password = "nm13j4d25h";

    PersoaneRepository persoaneRepository = new PersoaneRepository(url, userName, password);
    NevoiRepository nevoiRepository = new NevoiRepository(url, userName, password);

    Service service = new Service(persoaneRepository, nevoiRepository);

    @Override
    public void start(Stage stage) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(faptebune.Main.class.getResource("logInSignUpView.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 500, 370);
        LogInSignUpController controller = fxmlLoader.getController();
        controller.initialise(service);
        stage.setTitle("Fapte Bune");
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}