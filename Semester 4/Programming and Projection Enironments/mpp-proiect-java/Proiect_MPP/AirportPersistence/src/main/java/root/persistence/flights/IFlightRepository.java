package root.persistence.flights;

import root.model.Flight;
import root.persistence.Repository;

import java.time.LocalDateTime;
import java.util.Collection;

public interface IFlightRepository extends Repository<Flight, Integer> {
    Collection<Flight> getByDepartureAirport(int departureAirport);
    Collection<Flight> getByDestinationAirport(int destinationAirport);
    Collection<Flight> getAfterDepartureDateTime(LocalDateTime departureDateTime);
    Collection<Flight> getAvailable();
}
