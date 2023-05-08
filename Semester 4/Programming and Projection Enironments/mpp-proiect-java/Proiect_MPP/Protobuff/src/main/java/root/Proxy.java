package root;

import root.model.Airport;
import root.model.DTODestinationDate;
import root.model.DTOFlight;
import root.model.Flight;
import root.model.people.Client;
import root.model.people.Employee;
import root.model.people.Person;
import root.services.IObserver;
import root.services.IService;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.time.LocalDate;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class Proxy implements IService {
    private String host;
    private int port;
    private IObserver client;
    private InputStream input;
    private OutputStream output;
    private Socket connection;
    private BlockingQueue<Protobuffs.Response> qresponses;
    private volatile boolean finished;

    public Proxy(String host, int port) {
        this.host = host;
        this.port = port;
        qresponses = new LinkedBlockingQueue<>();
    }

    private void initConnection() throws Exception {
        connection = new Socket(host, port);
        output = connection.getOutputStream();
        input = connection.getInputStream();
        finished = false;
        startReader();
    }

    private void closeConnection() {
        finished = true;
        try {
            input.close();
            output.close();
            connection.close();
            client = null;

        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private boolean isUpdate(Protobuffs.Response.Type response) {
        return response == Protobuffs.Response.Type.TICKET_BOUGHT;
    }

    private void handleUpdate(Protobuffs.Response response) {
        Collection<Flight> flights = Utils.getFlights(response);
        try {
            client.ticketBought(flights);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private class ReaderThread implements Runnable {
        @Override
        public void run() {
            while (!finished) {
                try {
                    Protobuffs.Response response = Protobuffs.Response.parseDelimitedFrom(input);
                    System.out.println("Response received" + response);
                    if (isUpdate(response.getType())) {
                        handleUpdate(response);
                    } else {
                        try {
                            qresponses.put(response);
                        } catch (InterruptedException ex) {
                            System.out.println("Error " + ex);
                        }
                    }

                } catch (IOException ex) {
                    //ex.printStackTrace();
                    System.out.println("Reading error " + ex);
                }
            }
        }
    }

    private void startReader() {
        Thread tw = new Thread(new ReaderThread());
        tw.start();
    }

    private void sendRequest(Protobuffs.Request request) throws Exception {
        System.out.println("Sending request " + request);
        try {
            request.writeDelimitedTo(output);
            output.flush();
        } catch (IOException ex) {
            throw new Exception("Error sending object " + ex);
        }
    }

    private Protobuffs.Response readResponse() throws Exception {
        Protobuffs.Response response = null;
        try {
            response = qresponses.take();
        } catch (InterruptedException ex) {
            ex.printStackTrace();
        }
        return response;
    }

    public Employee login(Employee employee, IObserver client) throws Exception {
        initConnection();
        Protobuffs.Request req = Utils.createLoginRequest(employee);
        sendRequest(req);
        Protobuffs.Response response = readResponse();
        if (response.getType() == Protobuffs.Response.Type.OK) {
            this.client = client;
            Employee employeeToConnect = Utils.getEmployee(response);
            return employeeToConnect;
        }
        if (response.getType() == Protobuffs.Response.Type.ERROR) {
            String err = Utils.getError(response);
            closeConnection();
            throw new Exception(err);
        }
        return null;
    }

    @Override
    public void logout(Employee employee, IObserver client) throws Exception {
        Protobuffs.Request req = Utils.createLogoutRequest(employee);
        sendRequest(req);
        Protobuffs.Response response = readResponse();
        closeConnection();
        if (response.getType() == Protobuffs.Response.Type.ERROR) {
            String err = Utils.getError(response);
            throw new Exception(err);
        } else {
            System.out.println("Logout successful");
        }
    }

    @Override
    public Collection<Flight> findFlightByDestinationDate(String destination, LocalDate date) throws Exception {
        DTODestinationDate destinationDate = new DTODestinationDate(destination, date);
        Protobuffs.Request request = Utils.createFlightByDestinationDateRequest(destinationDate);
        sendRequest(request);
        Protobuffs.Response response = readResponse();
        if (response.getType() == Protobuffs.Response.Type.GET_DD_FLIGHT) {
            return Utils.getFlights(response);
        } else
            throw new Exception("Error getting flights");
    }

    @Override
    public Airport findAirportById(int id) throws Exception {
        Protobuffs.Request request = Utils.createFindAirportByIdRequest(id);
        sendRequest(request);
        Protobuffs.Response response = readResponse();
        if (response.getType() == Protobuffs.Response.Type.GET_AIRPORT) {
            return Utils.getAirport(response);
        } else
            throw new Exception("Error getting airport");
    }

    @Override
    public Flight findFlightById(int id) throws Exception {
        Protobuffs.Request request = Utils.createFindFlightByIdRequest(id);
        sendRequest(request);
        Protobuffs.Response response = readResponse();
        if (response.getType() == Protobuffs.Response.Type.GET_FLIGHT) {
            return Utils.getFlight(response);
        } else
            throw new Exception("Error getting airport");
    }

    @Override
    public Collection<Flight> getAllAvailableFlights() throws Exception {
        Protobuffs.Request request = Utils.createGetAllAvailableFlightsRequest();
        sendRequest(request);
        Protobuffs.Response response = readResponse();
        if (response.getType() == Protobuffs.Response.Type.GET_A_FLIGHTS) {
            return Utils.getFlights(response);
        } else
            throw new Exception("Error getting flights");
    }

    @Override
    public Collection<Airport> getAllAirports() throws Exception {
        Protobuffs.Request request = Utils.createGetAllAirportsRequest();
        sendRequest(request);
        Protobuffs.Response response = readResponse();
        if (response.getType() == Protobuffs.Response.Type.GET_ALL_AIRPORTS) {
            return Utils.getAirports(response);
        } else
            throw new Exception("Error getting airports");
    }

    @Override
    public void buyTicket(Client client, List<Person> people, Flight flight) throws Exception {
        DTOFlight dtoFlight = new DTOFlight(client, people, flight);
        Protobuffs.Request request = Utils.createBuyTicketRequest(dtoFlight);
        sendRequest(request);
        Protobuffs.Response response = readResponse();
        if (response.getType() == Protobuffs.Response.Type.OK) {
            System.out.println("Ticket bought successfully");
        } else
            throw new Exception("Error buying ticket");
    }
}
