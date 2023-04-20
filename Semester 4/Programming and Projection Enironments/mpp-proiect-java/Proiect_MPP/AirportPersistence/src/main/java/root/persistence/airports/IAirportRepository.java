package root.persistence.airports;

import root.model.Airport;
import root.persistence.Repository;

import java.util.Collection;

public interface IAirportRepository extends Repository<Airport, Integer> {
    Collection<Airport> getAirportAfterName(String name);
    Collection<Airport> getAirportsInCity(String cityName);
}
