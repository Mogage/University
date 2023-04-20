package root.client.gui;

import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.stage.Stage;
import javafx.util.StringConverter;
import root.client.StartRpcClient;
import root.model.*;
import root.model.people.Employee;
import root.services.IObserver;
import root.services.IService;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Properties;

public class MainController implements IObserver {
    @FXML
    public TextField destinationSearchInput;
    @FXML
    public DatePicker dateSearchInput;
    @FXML
    public Button ticketsButton;
    @FXML
    public Button logOutButton;
    @FXML
    public TableView<DTOAirportFlight> flightsTable;
    @FXML
    public TableColumn<DTOAirportFlight, String> departureCityColumn;
    @FXML
    public TableColumn<DTOAirportFlight, String> departureAirportColumn;
    @FXML
    public TableColumn<DTOAirportFlight, String> destinationCityColumn;
    @FXML
    public TableColumn<DTOAirportFlight, String> destinationAirportColumn;
    @FXML
    public TableColumn<DTOAirportFlight, String> departureDateColumn;
    @FXML
    public TableColumn<DTOAirportFlight, String> departureTimeColumn;
    @FXML
    public TableColumn<DTOAirportFlight, Integer> freeSeatsColumn;
    @FXML
    public TableView<DTOAirportFlight> searchFlightsTable;
    @FXML
    public TableColumn<DTOAirportFlight, String> searchDepartureCityColumn;
    @FXML
    public TableColumn<DTOAirportFlight, String> searchDepartureAirportColumn;
    @FXML
    public TableColumn<DTOAirportFlight, String> searchDestinationCityColumn;
    @FXML
    public TableColumn<DTOAirportFlight, String> searchDestinationAirportColumn;
    @FXML
    public TableColumn<DTOAirportFlight, String> searchDepartureDateColumn;
    @FXML
    public TableColumn<DTOAirportFlight, String> searchDepartureTimeColumn;
    @FXML
    public TableColumn<DTOAirportFlight, Integer> searchFreeSeatsColumn;

    private final ObservableList<DTOAirportFlight> flightsList = FXCollections.observableArrayList();
    private final ObservableList<DTOAirportFlight> searchList = FXCollections.observableArrayList();

    private IService service;
    private Stage logInStage;
    private Employee loggedUser;
    private Parent root;
    private BuyTicketController buyTicketController;
    private final Stage stage = new Stage();

    public void setServer(IService service) {
        this.service = service;
    }

    public void setStage(Stage stage) {
        this.logInStage = stage;
    }

    public void setLoggedUser(Employee loggedUser) {
        this.loggedUser = loggedUser;
    }

    public void setParent(Parent root) {
        this.root = root;
    }

    public void setBuyTicketController(BuyTicketController buyTicketController) {
        this.buyTicketController = buyTicketController;
    }

    public void initialise() {
        ticketsButton.disableProperty().bind(Bindings.isEmpty(searchFlightsTable.getSelectionModel().getSelectedItems()));
        destinationSearchInput.textProperty().addListener(o -> updateSearchTable());
        dateSearchInput.valueProperty().addListener(o -> updateSearchTable());
        dateSearchInput.setValue(LocalDate.now());
        dateSearchInput.setConverter(new StringConverter<>() {
            @Override
            public String toString(LocalDate date) {
                if (date != null) {
                    return Constants.DATE_FORMATTER.format(date);
                } else {
                    return "";
                }
            }

            @Override
            public LocalDate fromString(String string) {
                if (string != null && !string.isEmpty()) {
                    return LocalDate.parse(string, Constants.DATE_FORMATTER);
                } else {
                    return null;
                }
            }
        });

        initialiseTables();
    }

    private void initialiseTables() {
        initialiseFlightsTable();
        initialiseSearchTable();
    }

    private void initialiseFlightsTable() {
        departureCityColumn.setCellValueFactory(new PropertyValueFactory<>("departureCityName"));
        departureAirportColumn.setCellValueFactory(new PropertyValueFactory<>("departureName"));
        destinationCityColumn.setCellValueFactory(new PropertyValueFactory<>("destinationCityName"));
        destinationAirportColumn.setCellValueFactory(new PropertyValueFactory<>("destinationName"));
        departureDateColumn.setCellValueFactory(new PropertyValueFactory<>("departureDate"));
        departureTimeColumn.setCellValueFactory(new PropertyValueFactory<>("departureTime"));
        freeSeatsColumn.setCellValueFactory(new PropertyValueFactory<>("freeSeats"));
        updateFlightsTable();
    }

    private void initialiseSearchTable() {
        searchDepartureCityColumn.setCellValueFactory(new PropertyValueFactory<>("departureCityName"));
        searchDepartureAirportColumn.setCellValueFactory(new PropertyValueFactory<>("departureName"));
        searchDestinationCityColumn.setCellValueFactory(new PropertyValueFactory<>("destinationCityName"));
        searchDestinationAirportColumn.setCellValueFactory(new PropertyValueFactory<>("destinationName"));
        searchDepartureDateColumn.setCellValueFactory(new PropertyValueFactory<>("departureDate"));
        searchDepartureTimeColumn.setCellValueFactory(new PropertyValueFactory<>("departureTime"));
        searchFreeSeatsColumn.setCellValueFactory(new PropertyValueFactory<>("freeSeats"));
        updateSearchTable();
    }

    private List<DTOAirportFlight> updateLists(Collection<Flight> flights) {
        List<DTOAirportFlight> dtoAirportFlightList = new ArrayList<>();
        Airport departure;
        Airport destination;

        for (Flight flight : flights) {
            try {
                departure = service.findAirportById(flight.getDepartureAirport());
                destination = service.findAirportById(flight.getDestinationAirport());
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
            dtoAirportFlightList.add(new DTOAirportFlight(flight.getId(), departure.getCityName(), departure.getName(),
                    destination.getCityName(), destination.getName(), flight.getDepartureDate(), flight.getDepartureTime(),
                    flight.getFreeSeats()));
        }

        return dtoAirportFlightList;
    }

    private void updateFlightsTable() {
        Collection<Flight> flights;
        try {
            flights = service.getAllAvailableFlights();
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }
        flightsList.setAll(updateLists(flights));
        flightsTable.setItems(flightsList);
    }

    public void updateFlights(int flightId, int numberOfSeats) {
        Flight flight;
        try {
            flight = service.findFlightById(flightId);
            flight.setFreeSeats(numberOfSeats);
            service.updateFlight(flight, flightId);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        updateFlightsTable();
        updateSearchTable();
    }

    private void updateSearchTable() {
        String destination = destinationSearchInput.getText();
        LocalDate date = dateSearchInput.getValue();
        Collection<Flight> flights;
        try {
            flights = service.findFlightByDestinationDate(destination, date);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }
        searchList.setAll(updateLists(flights));
        searchFlightsTable.setItems(searchList);
    }

    @FXML
    public void buyTicketAction() {
        DTOAirportFlight airportFlight = searchFlightsTable.getSelectionModel().getSelectedItem();
        destinationSearchInput.setText("");
        dateSearchInput.setValue(LocalDate.now());
        buyTicketController.setServer(service);
        buyTicketController.setAirportFlight(airportFlight);
        buyTicketController.setMainController(this);

        //controller.initialise(properties, airportFlight, this);
        stage.setTitle("Ticket");
        if (stage.getScene() == null)
            stage.setScene(new Scene(root, 600, 300));
        //stage.setOnCloseRequest(event -> buy);

        stage.show();
    }

    @FXML
    public void logOutAction() {
        try {
            service.logout(loggedUser, this);
            logInStage.show();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void ticketBought(Ticket ticket) {
        Platform.runLater(() -> {
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Ticket bought");
            alert.setHeaderText("Ticket bought");
            alert.setContentText("Ticket bought successfully!");
            alert.showAndWait();
        });
    }
}