package root.proiect_mpp.controllers;

import javafx.beans.binding.Bindings;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.stage.Stage;
import javafx.util.StringConverter;
import root.proiect_mpp.Main;
import root.proiect_mpp.domain.Airport;
import root.proiect_mpp.domain.DTOAirportFlight;
import root.proiect_mpp.domain.Flight;
import root.proiect_mpp.repositories.airports.AirportRepository;
import root.proiect_mpp.repositories.flights.FlightRepository;
import root.proiect_mpp.service.main.MainService;
import root.proiect_mpp.utils.Constants;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Properties;

public class MainController {
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

    private Properties properties;
    private MainService mainService;

    public void initialise(Properties properties) {
        this.properties = properties;

        mainService = new MainService(new FlightRepository(properties), new AirportRepository(properties));
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
            departure = mainService.findAirportById(flight.getDepartureAirport());
            destination = mainService.findAirportById(flight.getDestinationAirport());
            dtoAirportFlightList.add(new DTOAirportFlight(flight.getId(), departure.getCityName(), departure.getName(),
                    destination.getCityName(), destination.getName(), flight.getDepartureDate(), flight.getDepartureTime(),
                    flight.getFreeSeats()));
        }

        return dtoAirportFlightList;
    }

    private void updateFlightsTable() {
        Collection<Flight> flights = mainService.getAllAvailableFlights();
        flightsList.setAll(updateLists(flights));
        flightsTable.setItems(flightsList);
    }

    private void updateSearchTable() {
        String destination = destinationSearchInput.getText();
        LocalDate date = dateSearchInput.getValue();
        Collection<Flight> flights = mainService.findByDestinationDate(destination, date);
        searchList.setAll(updateLists(flights));
        searchFlightsTable.setItems(searchList);
    }

    public void updateFlights(int flightId, int numberOfSeats) {
        Flight flight = mainService.findFlightById(flightId);
        flight.setFreeSeats(numberOfSeats);
        mainService.updateFlight(flight, flightId);
        updateFlightsTable();
        updateSearchTable();
    }

    @FXML
    public void buyTicketAction() {
        Scene scene;
        FXMLLoader fxmlLoader = new FXMLLoader(Main.class.getResource("BuyTicketView.fxml"));

        try {
            scene = new Scene(fxmlLoader.load(), 600, 300);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        DTOAirportFlight airportFlight = searchFlightsTable.getSelectionModel().getSelectedItem();
        destinationSearchInput.setText("");
        dateSearchInput.setValue(LocalDate.now());
        BuyTicketController controller = fxmlLoader.getController();
        controller.initialise(properties, airportFlight, this);

        Stage newStage = new Stage();
        newStage.setScene(scene);
        newStage.setResizable(false);
        newStage.setTitle("Buy Tickets");
        newStage.show();
    }

    @FXML
    private void logOutAction() {
        Scene scene;
        FXMLLoader fxmlLoader = new FXMLLoader(Main.class.getResource("LogInView.fxml"));

        try {
            scene = new Scene(fxmlLoader.load(), 200, 250);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        LogInController mainController = fxmlLoader.getController();
        mainController.initialise(properties);

        Stage currentStage = (Stage) logOutButton.getScene().getWindow();

        Stage newStage = new Stage();
        newStage.setScene(scene);
        newStage.setResizable(false);
        newStage.setTitle("Log In");
        currentStage.close();
        newStage.show();
    }
}