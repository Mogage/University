package root.client.gui;

import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
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
    public Button searchButton;
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
    private Collection<Airport> airports = null;

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
        Collection<Flight> flights = null;
        Collection<Flight> availableFlights = null;
        try {
            flights = service.findFlightByDestinationDate("", LocalDate.now());
            availableFlights = service.getAllAvailableFlights();
            airports = service.getAllAirports();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        Collection<Flight> finalFlights = flights;
        Collection<Flight> finalAvailableFlights = availableFlights;

        ticketsButton.disableProperty().bind(Bindings.isEmpty(searchFlightsTable.getSelectionModel().getSelectedItems()));
        //destinationSearchInput.textProperty().addListener(o -> updateSearchTable(finalAirports));
        //dateSearchInput.valueProperty().addListener(o -> updateSearchTable(finalAirports));
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

        initialiseFlightsTable(finalAvailableFlights);
        initialiseSearchTable(finalFlights);
    }


    private void initialiseFlightsTable(Collection<Flight> flights) {
        departureCityColumn.setCellValueFactory(new PropertyValueFactory<>("departureCityName"));
        departureAirportColumn.setCellValueFactory(new PropertyValueFactory<>("departureName"));
        destinationCityColumn.setCellValueFactory(new PropertyValueFactory<>("destinationCityName"));
        destinationAirportColumn.setCellValueFactory(new PropertyValueFactory<>("destinationName"));
        departureDateColumn.setCellValueFactory(new PropertyValueFactory<>("departureDate"));
        departureTimeColumn.setCellValueFactory(new PropertyValueFactory<>("departureTime"));
        freeSeatsColumn.setCellValueFactory(new PropertyValueFactory<>("freeSeats"));
        updateFlightsTable(flights);
    }

    private void initialiseSearchTable(Collection<Flight> flights) {
        searchDepartureCityColumn.setCellValueFactory(new PropertyValueFactory<>("departureCityName"));
        searchDepartureAirportColumn.setCellValueFactory(new PropertyValueFactory<>("departureName"));
        searchDestinationCityColumn.setCellValueFactory(new PropertyValueFactory<>("destinationCityName"));
        searchDestinationAirportColumn.setCellValueFactory(new PropertyValueFactory<>("destinationName"));
        searchDepartureDateColumn.setCellValueFactory(new PropertyValueFactory<>("departureDate"));
        searchDepartureTimeColumn.setCellValueFactory(new PropertyValueFactory<>("departureTime"));
        searchFreeSeatsColumn.setCellValueFactory(new PropertyValueFactory<>("freeSeats"));
        updateSearchTable(flights);
    }

    private List<DTOAirportFlight> updateLists(Collection<Flight> flights) {
        List<DTOAirportFlight> dtoAirportFlightList = new ArrayList<>();
        List<Airport> airportsList = airports.stream().toList();
        Airport departure;
        Airport destination;

        for (Flight flight : flights) {
            departure = airportsList.stream().filter(airport -> airport.getId() == flight.getDepartureAirport()).findFirst().orElse(null);
            destination = airportsList.stream().filter(airport -> airport.getId() == flight.getDestinationAirport()).findFirst().orElse(null);
            dtoAirportFlightList.add(new DTOAirportFlight(flight.getId(), departure.getCityName(), departure.getName(),
                    destination.getCityName(), destination.getName(), flight.getDepartureDate(), flight.getDepartureTime(),
                    flight.getFreeSeats()));
        }

        return dtoAirportFlightList;
    }

    private void updateFlightsTable(Collection<Flight> flights) {
        flightsList.setAll(updateLists(flights));
        flightsTable.setItems(flightsList);
    }

//    public void updateFlights(int flightId, int numberOfSeats) {
//        Flight flight;
//        Collection<Flight> flights = null;
//        Collection<Flight> availableFlights = null;
//        Collection<Airport> airports = null;
//        try {
//            //flight = service.findFlightById(flightId);
//            //flight.setFreeSeats(numberOfSeats);
//           // service.updateFlight(flightId, numberOfSeats);
//            flights = service.findFlightByDestinationDate(destinationSearchInput.getText(), dateSearchInput.getValue());
//            availableFlights = service.getAllAvailableFlights();
//            airports = service.getAllAirports();
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        }
//        Collection<Flight> finalFlights = flights;
//        Collection<Flight> finalAvailableFlights = availableFlights;
//        Collection<Airport> finalAirports = airports;
//        updateFlightsTable(finalAvailableFlights, finalAirports);
//        updateSearchTable(finalFlights, finalAirports);
//    }

    private void updateSearchTable(Collection<Flight> flights) {
        searchList.setAll(updateLists(flights));
        searchFlightsTable.setItems(searchList);
    }

    @FXML
    public void buyTicketAction() {
        DTOAirportFlight airportFlight = searchFlightsTable.getSelectionModel().getSelectedItem();
        destinationSearchInput.setText("");
        dateSearchInput.setValue(LocalDate.now());
        updateSearchTable(new ArrayList<>());
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
            stage.close();
            service.logout(loggedUser, this);
            logInStage.show();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void ticketBought(Collection<Flight> flights) {
        Platform.runLater(() -> {
            try {
                updateFlightsTable(flights);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    public void searchAction() {
        String destination = destinationSearchInput.getText();
        LocalDate date = dateSearchInput.getValue();
        try {
            updateSearchTable(service.findFlightByDestinationDate(destination, date));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}