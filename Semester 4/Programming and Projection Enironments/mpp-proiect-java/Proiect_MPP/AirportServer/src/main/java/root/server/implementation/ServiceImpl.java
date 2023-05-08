package root.server.implementation;

import root.model.Airport;
import root.model.Flight;
import root.model.Invoice;
import root.model.Ticket;
import root.model.people.Client;
import root.model.people.Employee;
import root.model.people.Person;
import root.persistence.airports.IAirportRepository;
import root.persistence.flights.IFlightRepository;
import root.persistence.invoices.IInvoiceRepository;
import root.persistence.people.clients.IClientRepository;
import root.persistence.people.employees.IEmployeeRepository;
import root.persistence.tickets.ITicketRepository;
import root.services.IObserver;
import root.services.IService;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ServiceImpl implements IService {
    IEmployeeRepository employeeRepository;
    IFlightRepository flightRepository;
    IInvoiceRepository invoiceRepository;
    ITicketRepository ticketRepository;
    IAirportRepository airportRepository;
    IClientRepository clientRepository;

    private final Map<Integer, IObserver> loggedEmployees;

    public ServiceImpl(IEmployeeRepository employeeRepository, IFlightRepository flightRepository,
                       IInvoiceRepository invoiceRepository, ITicketRepository ticketRepository,
                       IAirportRepository airportRepository, IClientRepository clientRepository) {
        this.employeeRepository = employeeRepository;
        this.flightRepository = flightRepository;
        this.invoiceRepository = invoiceRepository;
        this.ticketRepository = ticketRepository;
        this.airportRepository = airportRepository;
        this.clientRepository = clientRepository;

        loggedEmployees = new ConcurrentHashMap<>();
    }

    @Override
    public synchronized Employee login(Employee employee, IObserver client) throws Exception {
        Employee employeeToLogin = employeeRepository.findByEmail(employee.getEmail());
        if (employeeToLogin != null) {
            if (loggedEmployees.get(employeeToLogin.getId()) != null)
                throw new Exception("Employee already logged in.");

            loggedEmployees.put(employeeToLogin.getId(), client);
            return employeeToLogin;
        } else {
            throw new Exception("Authentication failed.");
        }
    }

    @Override
    public synchronized void logout(Employee employee, IObserver client) throws Exception {
        IObserver loggedEmployee = loggedEmployees.remove(employee.getId());
        if (loggedEmployee == null) {
            throw new Exception("Employee " + employee.getId().toString() + " is not logged in.");
        }
    }

    @Override
    public synchronized Collection<Flight> findFlightByDestinationDate(String destination, LocalDate date) {
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

//    @Override
//    public void updateFlight(int id, int numberOfSeats) {
//        Flight flight = flightRepository.findById(id);
//        flight.setFreeSeats(numberOfSeats);
//        flightRepository.update(flight, id);
//    }

    @Override
    public synchronized Airport findAirportById(int id) {
        return airportRepository.findById(id);
    }

    @Override
    public synchronized Flight findFlightById(int id) {
        return flightRepository.findById(id);
    }

    @Override
    public synchronized Collection<Flight> getAllAvailableFlights() {
        return flightRepository.getAvailable();
    }

    @Override
    public synchronized Collection<Airport> getAllAirports() {
        return airportRepository.getAll();
    }

    @Override
    public synchronized void buyTicket(Client client, List<Person> people, Flight flight) throws Exception {
        if (client.getFirstName().isBlank() || client.getLastName().isBlank() || client.getAddress().isBlank()) {
            throw new Exception("Client can't be empty");
        }
        if (null == clientRepository.findByAddress(client.getAddress())) {
            clientRepository.add(client);
        }
        int numberOfTickets = 1;
        for (Person person : people) {
            if (person.getFirstName().isBlank() || person.getLastName().isBlank()) {
                break;
            }
            numberOfTickets = numberOfTickets + 1;
        }
        if (numberOfTickets > flight.getFreeSeats()) {
            throw new Exception("There are not this many free seats.");
        }

        Invoice invoice = new Invoice(clientRepository.findByAddress(client.getAddress()).getId());
        invoice.setId(invoiceRepository.add(invoice));
        // addTicket(flight.getId(), invoice.getId(), client.getFirstName() + " " + client.getLastName());
        Ticket ticket = new Ticket(flight.getId(), invoice.getId(), 1, client.getFirstName() + " " + client.getLastName());
        List<Ticket> tickets = new ArrayList<>();
        tickets.add(ticket);
        ticketRepository.add(ticket);
        for (Person person : people) {
            if (person.getFirstName().isBlank() || person.getLastName().isBlank()) {
                break;
            }
            // addTicket(flight.getId(), invoice.getId(), person.getFirstName() + " " + person.getLastName());
            ticket = new Ticket(flight.getId(), invoice.getId(), 1, client.getFirstName() + " " + client.getLastName());
            ticketRepository.add(ticket);
            tickets.add(ticket);
        }

        Flight newFlight = new Flight(flight.getId(), flight.getFreeSeats() - numberOfTickets, flight.getDestinationAirport(),
                flight.getDepartureAirport(), flight.getDepartureDate(), flight.getDepartureTime());
        flightRepository.update(newFlight, flight.getId());

        for (IObserver loggedEmployee : loggedEmployees.values()) {
            loggedEmployee.ticketBought(flightRepository.getAvailable());
        }

        //return numberOfTickets;
    }
}
