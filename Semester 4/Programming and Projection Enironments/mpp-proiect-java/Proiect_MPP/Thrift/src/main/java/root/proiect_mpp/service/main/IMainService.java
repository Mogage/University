package root.proiect_mpp.service.main;

import root.proiect_mpp.domain.Airport;
import root.proiect_mpp.domain.Flight;

import java.time.LocalDate;
import java.util.Collection;

public interface IMainService {
    void updateFlight(Flight elem, int id);

    Collection<Flight> findByDestinationDate(String destination, LocalDate date);

    Airport findAirportById(int id);

    Flight findFlightById(int id);

    Collection<Flight> getAllAvailableFlights();
}
