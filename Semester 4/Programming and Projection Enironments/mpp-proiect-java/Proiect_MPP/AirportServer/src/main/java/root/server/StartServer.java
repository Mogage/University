package root.server;

import root.persistence.airports.AirportRepository;
import root.persistence.airports.IAirportRepository;
import root.persistence.flights.FlightRepository;
import root.persistence.flights.IFlightRepository;
import root.persistence.invoices.IInvoiceRepository;
import root.persistence.invoices.InvoiceRepository;
import root.persistence.people.clients.ClientRepository;
import root.persistence.people.clients.IClientRepository;
import root.persistence.people.employees.EmployeeRepository;
import root.persistence.people.employees.IEmployeeRepository;
import root.persistence.tickets.ITicketRepository;
import root.persistence.tickets.TicketRepository;
import root.server.implementation.ServiceImpl;
import root.services.IService;
import root.utils.AbstractServer;
import root.utils.RpcConcurrentServer;
import root.utils.ServerException;

import java.io.IOException;
import java.util.Properties;

public class StartServer {
    private final static int defaultPort = 55555;

    public static void main(String[] args) {
        Properties serverProps = new Properties();
        try {
            serverProps.load(StartServer.class.getResourceAsStream("/bd.config"));
            System.out.println("Server properties set. ");
            serverProps.list(System.out);
        } catch (IOException e) {
            System.err.println("Cannot find properties " + e);
            return;
        }


        IEmployeeRepository employeeRepository = new EmployeeRepository(serverProps);
        ITicketRepository ticketRepository = new TicketRepository(serverProps);
        IFlightRepository flightRepository = new FlightRepository(serverProps);
        IAirportRepository airportRepository = new AirportRepository(serverProps);
        IInvoiceRepository invoiceRepository = new InvoiceRepository(serverProps);
        IClientRepository clientRepository = new ClientRepository(serverProps);

        IService service = new ServiceImpl(employeeRepository, flightRepository, invoiceRepository, ticketRepository,
                airportRepository, clientRepository);

        int ServerPort = defaultPort;
        try {
            ServerPort = Integer.parseInt(serverProps.getProperty("server.port"));
        } catch (NumberFormatException nef) {
            System.err.println("Wrong  Port Number" + nef.getMessage());
            System.err.println("Using default port " + defaultPort);
        }
        System.out.println("Starting server on port: " + ServerPort);
        AbstractServer server = new RpcConcurrentServer(ServerPort, service);
        try {
            server.start();
        } catch (ServerException e) {
            System.err.println("Error starting the server" + e.getMessage());
        } finally {
            try {
                server.stop();
            } catch (ServerException e) {
                System.err.println("Error stopping server " + e.getMessage());
            }
        }
    }
}
