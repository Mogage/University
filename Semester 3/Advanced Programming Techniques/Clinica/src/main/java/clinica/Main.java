package clinica;

import clinica.controllers.MainController;
import clinica.controllers.MedicController;
import clinica.domain.Consultatie;
import clinica.domain.Medic;
import clinica.domain.Sectie;
import clinica.repository.ConsultatiiRepository;
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
    Repository<Long, Consultatie> consultatieRepository = new ConsultatiiRepository(url, userName, password);

    MainService service = new MainService(sectieRepository, medicRepository, consultatieRepository);

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

        Iterable<Medic> medici = service.getMedici();
        for(Medic medic : medici) {
            FXMLLoader fxmlLoader1 = new FXMLLoader(Main.class.getResource("MedicView.fxml"));
            Scene scene1 = new Scene(fxmlLoader1.load(), 545, 255);
            MedicController medicController = fxmlLoader1.getController();
            medicController.initialise(service, medic);
            service.addObserver(medicController);
            Stage stage1 = new Stage();
            stage1.setScene(scene1);
            stage1.setResizable(false);
            stage1.setTitle("Salut!");
            stage1.show();
        }

    }

    public static void main(String[] args) {
        launch();
    }
}