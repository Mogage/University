package root.proiect_mpp.service.main;

import root.proiect_mpp.domain.Airport;
import root.proiect_mpp.domain.Flight;
import root.proiect_mpp.repositories.airports.AirportRepository;
import root.proiect_mpp.repositories.flights.FlightRepository;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class MainService implements IMainService {

    private final FlightRepository flightRepository;
    private final AirportRepository airportRepository;

    public MainService(FlightRepository flightRepository, AirportRepository airportRepository) {
        this.flightRepository = flightRepository;
        this.airportRepository = airportRepository;
    }

    @Override
    public void updateFlight(Flight elem, int id) {
        flightRepository.update(elem, id);
    }

    @Override
    public Airport findAirportById(int id) {
        return airportRepository.findById(id);
    }

    @Override
    public Flight findFlightById(int id) {
        return flightRepository.findById(id);
    }

    @Override
    public Collection<Flight> getAllAvailableFlights() {
        return flightRepository.getAvailable();
    }

    @Override
    public Collection<Flight> findByDestinationDate(String destination, LocalDate date) {
        Collection<Airport> airports = airportRepository.getAirportsInCity(destination);
        List<Flight> flightList = new ArrayList<>();
        Collection<Flight> flights;
        LocalDate departureDate;
        for (Airport airport : airports) {
            flights = flightRepository.getByDestinationAirport(airport.getId());
            if (null == flights) {
                continue;
            }
            for (Flight flight : flights) {
                departureDate = flight.getDepartureDate();
                if (departureDate.getYear() != date.getYear() ||
                        departureDate.getMonthValue() != date.getMonthValue() ||
                        departureDate.getDayOfMonth() != date.getDayOfMonth() ||
                        flight.getFreeSeats() == 0) {
                    continue;
                }
                flightList.add(flight);
            }
        }
        return flightList;
    }
}
