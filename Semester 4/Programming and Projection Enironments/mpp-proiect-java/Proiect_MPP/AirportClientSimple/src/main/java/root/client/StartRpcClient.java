package root.client;

import root.client.gui.*;
import root.rpcprotocol.ClientRpcProxy;
import root.services.IService;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;
import java.util.Properties;


public class StartRpcClient extends Application {
    private final static int defaultChatPort = 55555;
    private final static String defaultServer = "localhost";

    public void start(Stage primaryStage) throws Exception {
        System.out.println("In start");
        Properties clientProps = new Properties();
        try {
            clientProps.load(StartRpcClient.class.getResourceAsStream("/client.properties"));
            System.out.println("Client properties set. ");
            clientProps.list(System.out);
        } catch (IOException e) {
            System.err.println("Cannot find client.properties " + e);
            return;
        }
        String serverIP = clientProps.getProperty("server.host", defaultServer);
        int serverPort = defaultChatPort;

        try {
            serverPort = Integer.parseInt(clientProps.getProperty("server.port"));
        } catch (NumberFormatException ex) {
            System.err.println("Wrong port number " + ex.getMessage());
            System.out.println("Using default port: " + defaultChatPort);
        }
        System.out.println("Using server IP " + serverIP);
        System.out.println("Using server port " + serverPort);

        IService server = new ClientRpcProxy(serverIP, serverPort);

        FXMLLoader loader = new FXMLLoader(getClass().getClassLoader().getResource("root/LogInView.fxml"));
        Parent root = loader.load();

        LogInController logInController = loader.getController();
        logInController.setServer(server);

        FXMLLoader mainLoader = new FXMLLoader(getClass().getClassLoader().getResource("root/MainView.fxml"));
        Parent mainRoot = mainLoader.load();

        MainController mainController = mainLoader.getController();
        mainController.initialise();
        mainController.setServer(server);

        logInController.setMainController(mainController);
        logInController.setParent(mainRoot);

        FXMLLoader ticketsLoader = new FXMLLoader(getClass().getClassLoader().getResource("root/BuyTicketView.fxml"));
        Parent ticketsRoot = ticketsLoader.load();

        BuyTicketController buyTicketController = ticketsLoader.getController();
        buyTicketController.setServer(server);

        mainController.setBuyTicketController(buyTicketController);
        mainController.setParent(ticketsRoot);

        primaryStage.setTitle("Log In");
        primaryStage.setScene(new Scene(root, 200, 250));
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}


