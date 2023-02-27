package examen;

import examen.controllers.LogInSignUpController;
import examen.domain.Nevoie;
import examen.domain.Persoana;
import examen.domain.validators.NevoieValidator;
import examen.domain.validators.PersoanaValidator;
import examen.domain.validators.Validator;
import examen.repository.NevoiRepository;
import examen.repository.PersoaneRepository;
import examen.service.Service;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

public class Main extends Application {
    String url = "jdbc:postgresql://localhost:5432/examen";
    String userName = "postgres";
    String password = "nm13j4d25h";

    Validator<Persoana> persoanaValidator = PersoanaValidator.getInstance();
    Validator<Nevoie> nevoieValidator = NevoieValidator.getInstance();

    PersoaneRepository persoaneRepository = new PersoaneRepository(url, userName, password);
    NevoiRepository nevoiRepository = new NevoiRepository(url, userName, password);

    Service service = new Service(persoanaValidator, nevoieValidator, persoaneRepository, nevoiRepository);

    @Override
    public void start(Stage stage) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(examen.Main.class.getResource("logInSignUpView.fxml"));
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