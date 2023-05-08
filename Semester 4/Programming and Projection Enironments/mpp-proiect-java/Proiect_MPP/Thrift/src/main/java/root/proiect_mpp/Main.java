package root.proiect_mpp;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;
import root.proiect_mpp.controllers.LogInController;

import java.io.FileReader;
import java.io.IOException;
import java.util.Properties;

public class Main extends Application {
    @Override
    public void start(Stage stage) throws IOException {
        Properties props = new Properties();
        props.load(new FileReader("bd.config"));

        FXMLLoader fxmlLoader = new FXMLLoader(Main.class.getResource("LogInView.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 200, 250);
        LogInController logInController = fxmlLoader.getController();
        logInController.initialise(props);
        stage.setTitle("Log In!");
        stage.setScene(scene);
        stage.setResizable(false);
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}