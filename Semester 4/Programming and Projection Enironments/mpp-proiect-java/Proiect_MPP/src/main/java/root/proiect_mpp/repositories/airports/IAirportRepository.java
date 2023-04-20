package root.proiect_mpp.repositories.airports;

import root.proiect_mpp.domain.Airport;
import root.proiect_mpp.repositories.Repository;

import java.util.Collection;

public interface IAirportRepository extends Repository<Airport, Integer> {
    Collection<Airport> getAirportAfterName(String name);
    Collection<Airport> getAirportsInCity(String cityName);
}
