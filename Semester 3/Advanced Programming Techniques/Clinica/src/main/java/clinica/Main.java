package clinica;

import clinica.controllers.MainController;
import clinica.domain.Medic;
import clinica.domain.Sectie;
import clinica.repository.MediciRepository;
import clinica.repository.Repository;
import clinica.repository.SectiiRepository;
import clinica.service.MainService;
import clinica.service.Service;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

public class Main extends Application {

    String url = "jdbc:postgresql://localhost:5432/clinica";
    String userName = "postgres";
    String password = "nm13j4d25h";

    Repository<Long, Sectie> sectieRepository = new SectiiRepository(url, userName, password);
    Repository<Long, Medic> medicRepository = new MediciRepository(url, userName, password);

    Service service = new MainService(sectieRepository, medicRepository);

    @Override
    public void start(Stage stage) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(Main.class.getResource("MainView.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 600, 400);
        MainController mainController = fxmlLoader.getController();
        mainController.initialise(service);
        stage.setTitle("Hello!");
        stage.setScene(scene);
        stage.setResizable(false);
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}