package root.proiect_mpp.repositories.flights;

import root.proiect_mpp.domain.Flight;
import root.proiect_mpp.repositories.Repository;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Collection;

public interface IFlightRepository extends Repository<Flight, Integer> {
    Collection<Flight> getByDepartureAirport(int departureAirport);
    Collection<Flight> getByDestinationAirport(int destinationAirport);
    Collection<Flight> getAfterDepartureDateTime(LocalDateTime departureDateTime);
    Collection<Flight> getAvailable();
}
