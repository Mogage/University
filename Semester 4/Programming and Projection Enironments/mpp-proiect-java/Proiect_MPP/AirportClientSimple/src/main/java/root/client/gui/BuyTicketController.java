package root.client.gui;

import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.TextField;
import root.model.DTOAirportFlight;
import root.model.people.Client;
import root.model.people.Person;
import root.services.IService;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

public class BuyTicketController {
    @FXML
    public Button buyTicketButton;
    @FXML
    public TextField clientFirstNameText;
    @FXML
    public TextField tourist1FirstNameText;
    @FXML
    public TextField tourist2FirstNameText;
    @FXML
    public TextField tourist3FirstNameText;
    @FXML
    public TextField tourist4FirstNameText;
    @FXML
    public TextField tourist5FirstNameText;
    @FXML
    public TextField clientAddressText;
    @FXML
    public TextField clientLastNameText;
    @FXML
    public TextField tourist1LastNameText;
    @FXML
    public TextField tourist2LastNameText;
    @FXML
    public TextField tourist3LastNameText;
    @FXML
    public TextField tourist4LastNameText;
    @FXML
    public TextField tourist5LastNameText;

    private DTOAirportFlight airportFlight;
    private IService service;
    private MainController mainController;

    public void setServer(IService service) {
        this.service = service;
    }
    public void setAirportFlight(DTOAirportFlight airportFlight) {
        this.airportFlight = airportFlight;
    }
    public void setMainController(MainController mainController) {
        this.mainController = mainController;
    }

    @FXML
    public void buyTicketAction() {
        Client client = new Client(clientFirstNameText.getText(), clientLastNameText.getText(), clientAddressText.getText());
        List<Person> people = new ArrayList<>();
        people.add(new Person(tourist1FirstNameText.getText(), tourist1LastNameText.getText()));
        people.add(new Person(tourist2FirstNameText.getText(), tourist2LastNameText.getText()));
        people.add(new Person(tourist3FirstNameText.getText(), tourist3LastNameText.getText()));
        people.add(new Person(tourist4FirstNameText.getText(), tourist4LastNameText.getText()));
        people.add(new Person(tourist5FirstNameText.getText(), tourist5LastNameText.getText()));

        try {
            service.buyTicket(client, people, airportFlight.toFlight());
            // int numberOfSeats = service.buyTicket(client, people, airportFlight.toFlight());
            // mainController.updateFlights(airportFlight.getId(), airportFlight.getFreeSeats() - numberOfSeats);
        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            alert.show();
        }
    }
}
