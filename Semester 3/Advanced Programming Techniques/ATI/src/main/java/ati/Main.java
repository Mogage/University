package ati;

import ati.controllers.AsteptariController;
import ati.controllers.PaturiController;
import ati.repository.PacientiRepository;
import ati.repository.PaturiRepository;
import ati.service.Service;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

public class Main extends Application {
    String url = "jdbc:postgresql://localhost:5432/clinica";
    String userName = "postgres";
    String password = "nm13j4d25h";

    PacientiRepository pacientiRepository = new PacientiRepository(url, userName, password);
    PaturiRepository paturiRepository = new PaturiRepository(url, userName, password);

    Service service = new Service(paturiRepository, pacientiRepository);

    @Override
    public void start(Stage stage) throws IOException {
        startPaturi(stage);
        startAsteptari();
    }

    private void startPaturi(Stage stage) throws IOException{
        FXMLLoader fxmlLoader = new FXMLLoader(Main.class.getResource("paturiView.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 163, 265);
        PaturiController paturiController = fxmlLoader.getController();
        paturiController.initialise(service);
        service.addObserver(paturiController);
        stage.setTitle("ATI");
        stage.setScene(scene);
        stage.show();
    }

    private void startAsteptari() throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(Main.class.getResource("asteptariView.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 380, 400);
        AsteptariController asteptariController = fxmlLoader.getController();
        asteptariController.initialise(service);
        Stage stage = new Stage();
        stage.setTitle("ATI");
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}