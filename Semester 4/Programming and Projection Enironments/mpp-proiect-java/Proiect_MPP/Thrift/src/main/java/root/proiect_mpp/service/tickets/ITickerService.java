package root.proiect_mpp.service.tickets;

import root.proiect_mpp.domain.DTOAirportFlight;
import root.proiect_mpp.domain.people.Client;
import root.proiect_mpp.domain.people.Person;

import java.util.List;

public interface ITickerService {
    int buyTicket(Client client, List<Person> people, DTOAirportFlight airportFlight) throws Exception;
}
