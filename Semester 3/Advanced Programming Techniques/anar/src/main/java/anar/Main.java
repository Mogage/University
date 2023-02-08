package anar;

import anar.controllers.AvertizariController;
import anar.controllers.RauriController;
import anar.repository.LocalitatiRepository;
import anar.repository.RauriRepository;
import anar.service.Service;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

public class Main extends Application {

    String url = "jdbc:postgresql://localhost:5432/clinica";
    String userName = "postgres";
    String password = "nm13j4d25h";

    RauriRepository rauriRepository = new RauriRepository(url, userName, password);
    LocalitatiRepository localitatiRepository = new LocalitatiRepository(url, userName, password);

    Service service = new Service(rauriRepository, localitatiRepository);

    @Override
    public void start(Stage stage) throws IOException {
        startRauri(stage);
        startLocalitati();
    }

    private void startRauri(Stage stage) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(Main.class.getResource("rauriView.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 270, 333);
        RauriController rauriController = fxmlLoader.getController();
        rauriController.initialise(service);
        service.addObserver(rauriController);
        stage.setTitle("Anar");
        stage.setScene(scene);
        stage.show();
    }

    private void startLocalitati() throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(Main.class.getResource("avertizariView.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 400, 350);
        AvertizariController avertizariController = fxmlLoader.getController();
        avertizariController.initialise(service);
        service.addObserver(avertizariController);
        Stage stage = new Stage();
        stage.setTitle("Anar");
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}