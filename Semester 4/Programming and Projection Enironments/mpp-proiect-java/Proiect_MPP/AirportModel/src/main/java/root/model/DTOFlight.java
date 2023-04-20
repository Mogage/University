package root.model;

import root.model.people.Client;
import root.model.people.Person;

import java.io.Serializable;
import java.util.List;

public class DTOFlight implements Serializable {
    private final Client client;
    private final List<Person> people;
    private final Flight flight;

    public DTOFlight(Client client, List<Person> people, Flight flight) {
        this.client = client;
        this.people = people;
        this.flight = flight;
    }

    public Client getClient() {
        return client;
    }

    public List<Person> getPeople() {
        return people;
    }

    public Flight getFlight() {
        return flight;
    }

    @Override
    public String toString() {
        return "DTOAirportFlight{" +
                "client=" + client +
                ", people=" + people +
                ", flight=" + flight +
                '}';
    }
}
