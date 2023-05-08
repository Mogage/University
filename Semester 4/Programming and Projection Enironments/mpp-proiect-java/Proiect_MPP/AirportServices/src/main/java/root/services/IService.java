package root.services;

import root.model.Airport;
import root.model.Flight;
import root.model.people.Client;
import root.model.people.Employee;
import root.model.people.Person;

import java.time.LocalDate;
import java.util.Collection;
import java.util.List;

public interface IService {
    // Employee findEmployeeByEmail(String email) throws Exception;
    Employee login(Employee employee, IObserver client) throws Exception;
    void logout(Employee employee, IObserver client) throws Exception;

    Collection<Flight> findFlightByDestinationDate(String destination, LocalDate date) throws Exception;
    Airport findAirportById(int id) throws Exception;
    Flight findFlightById(int id) throws Exception;
    Collection<Flight> getAllAvailableFlights() throws Exception;
    Collection<Airport> getAllAirports() throws Exception;

    //void updateFlight(int id, int numberOfSeats) throws Exception;

    void buyTicket(Client client, List<Person> people, Flight flight) throws Exception;
}
