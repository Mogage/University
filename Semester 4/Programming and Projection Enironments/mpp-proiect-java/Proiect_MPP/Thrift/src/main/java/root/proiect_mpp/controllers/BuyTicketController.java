package root.proiect_mpp.controllers;

import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.TextField;
import root.proiect_mpp.domain.DTOAirportFlight;
import root.proiect_mpp.domain.people.Client;
import root.proiect_mpp.domain.people.Person;
import root.proiect_mpp.repositories.invoices.InvoiceRepository;
import root.proiect_mpp.repositories.people.clients.ClientRepository;
import root.proiect_mpp.repositories.tickets.TicketRepository;
import root.proiect_mpp.service.tickets.TicketService;

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
    private TicketService ticketService;
    private MainController mainController;

    public void initialise(Properties properties, DTOAirportFlight airportFlight, MainController mainController) {
        ticketService = new TicketService(new TicketRepository(properties), new InvoiceRepository(properties), new ClientRepository(properties));
        this.airportFlight = airportFlight;
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
            int numberOfSeats = ticketService.buyTicket(client, people, airportFlight);
            mainController.updateFlights(airportFlight.getId(), airportFlight.getFreeSeats() - numberOfSeats);
        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            alert.show();
        }
    }
}
