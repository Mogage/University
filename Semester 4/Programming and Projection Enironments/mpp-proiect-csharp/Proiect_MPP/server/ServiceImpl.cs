using model;
using model.people;
using persistence.airports;
using persistence.flights;
using persistence.invoices;
using persistence.tickets;
using persistence.people.clients;
using persistence.people.employees;
using services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace server
{
    public class ServiceImpl : IService
    {
        private IEmployeeRepository employeeRepository;
        private IFlightRepository flightRepository;
        private IInvoiceRepository invoiceRepository;
        private ITicketRepository ticketRepository;
        private IAirportRepository airportRepository;
        private IClientRepository clientRepository;

        private readonly IDictionary<int, IObserver> loggedEmployees;

        public ServiceImpl(IEmployeeRepository employeeRepository, IFlightRepository flightRepository,
                       IInvoiceRepository invoiceRepository, ITicketRepository ticketRepository,
                       IAirportRepository airportRepository, IClientRepository clientRepository)
        {
            this.employeeRepository = employeeRepository;
            this.flightRepository = flightRepository;
            this.invoiceRepository = invoiceRepository;
            this.ticketRepository = ticketRepository;
            this.airportRepository = airportRepository;
            this.clientRepository = clientRepository;

            loggedEmployees = new Dictionary<int, IObserver>();
        }

        public Airport findAirportById(int id)
        {
            return airportRepository.findById(id);
        }

        public List<Flight> findFlightByDestinationDate(string destination, DateTime date)
        {
            List<Airport> airports = airportRepository.getAirportsInCity(destination);
            List<Flight> result = new List<Flight>();
            List<Flight> flights;
            DateTime departureDate;
            foreach (Airport airport in airports)
            {
                flights = flightRepository.getByDestinationAirport(airport.ID);
                if (flights == null)
                {
                    continue;
                }
                foreach (Flight flight in flights)
                {
                    departureDate = flight.DepartureDate;
                    if (flight.FreeSeats == 0 || 
                        date.Year != departureDate.Year || 
                        date.Month != departureDate.Month ||
                        date.Day != departureDate.Day)
                    {
                        continue;
                    }
                    result.Add(flight);
                }
            }
            return result;
        }

        public Flight findFlightById(int id)
        {
            return flightRepository.findById(id);
        }

        public List<Airport> getAllAirports()
        {
            return airportRepository.findAll();
        }

        public List<Flight> getAllAvailableFlights()
        {
            return flightRepository.getAvailable();
        }

        public Employee login(Employee employee, IObserver client)
        {
            Employee employeeToLogin = employeeRepository.findByEmail(employee.Email);
            if (employeeToLogin != null)
            {
                if (loggedEmployees.ContainsKey(employeeToLogin.ID))
                    throw new Exception("Employee already logged in.");

                loggedEmployees[employeeToLogin.ID] =  client;
                return employeeToLogin;
            }
            else
            {
                throw new Exception("Authentication failed.");
            }
        }

        public void logout(Employee employee, IObserver client)
        {
            if (loggedEmployees[employee.ID] == null)
            {
                throw new Exception("Employee " + employee.ID.ToString() + " is not logged in.");
            }
            loggedEmployees.Remove(employee.ID);
        }

        public void buyTicket(Client client, List<Person> people, Flight flight)
        {
            if (string.IsNullOrWhiteSpace(client.FirstName) || string.IsNullOrWhiteSpace(client.LastName) || string.IsNullOrWhiteSpace(client.Address))
            {
                throw new Exception("Client can't be empty.");
            }
            if (null == clientRepository.findByAddress(client.Address))
            {
                clientRepository.add(client);
            }
            int numberOfTickets = 1;
            foreach (Person person in people)
            {
                if (string.IsNullOrWhiteSpace(person.FirstName) || string.IsNullOrWhiteSpace(person.LastName))
                {
                    break;
                }
                numberOfTickets = numberOfTickets + 1;
            }

            if (numberOfTickets > flight.FreeSeats)
            {
                throw new Exception("There are not that many free seats.");
            }

            Invoice invoice = new Invoice(clientRepository.findByAddress(client.Address).ID);
            invoice.ID = invoiceRepository.add(invoice);
            Ticket ticket = new Ticket(flight.ID, invoice.ID, 1, client.FirstName + " " + client.LastName);
            List<Ticket> tickets = new List<Ticket> { ticket };
            ticketRepository.add(ticket);

            foreach (Person person in people)
            {
                if (string.IsNullOrWhiteSpace(person.FirstName) || string.IsNullOrWhiteSpace(person.LastName))
                {
                    break;
                }
                ticket = new Ticket(flight.ID, invoice.ID, 1, client.FirstName + " " + client.LastName);
                ticketRepository.add(ticket);
                tickets.Add(ticket);
            }

            Flight newFlight = new Flight(flight.ID, flight.FreeSeats - numberOfTickets, flight.DestinationAirport,
                flight.DepartureAirport, flight.DepartureDate, flight.DepartureTime);
            flightRepository.update(newFlight, flight.ID);

            foreach (IObserver loggedEmployee in loggedEmployees.Values)
            {
                loggedEmployee.ticketBought(flightRepository.getAvailable());
            }
        }
    }
}
