package root.rpcprotocol;

import root.model.*;
import root.model.people.Client;
import root.model.DTODestinationDate;
import root.model.people.Employee;
import root.model.people.Person;
import root.services.IObserver;
import root.services.IService;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.time.LocalDate;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class ClientRpcProxy implements IService {
    private final String host;
    private final int port;

    private IObserver client;

    private ObjectInputStream input;
    private ObjectOutputStream output;
    private Socket connection;

    private final BlockingQueue<Response> qresponses;
    private volatile boolean finished;

    public ClientRpcProxy(String host, int port) {
        this.host = host;
        this.port = port;
        qresponses = new LinkedBlockingQueue<>();
    }

    @Override
    public Employee login(Employee employee, IObserver client) throws Exception {
        initializeConnection();
        Request request = new Request.Builder().type(RequestType.LOGIN).data(employee).build();
        System.out.println("Sending request: " + request.data().toString());
        sendRequest(request);
        Response response = readResponse();
        switch (response.type()) {
            case OK -> {
                this.client = client;
                return (Employee) response.data();
            }
            case ERROR -> {
                String error = response.data().toString();
                closeConnection();
                throw new Exception(error);
            }
        }
        return null;
    }

    @Override
    public void logout(Employee employee, IObserver client) throws Exception {
        Request req = new Request.Builder().type(RequestType.LOGOUT).data(employee).build();
        sendRequest(req);
        Response response = readResponse();
        closeConnection();
        if (response.type() == ResponseType.ERROR) {
            String err = response.data().toString();
            throw new Exception(err);
        }
    }

    @Override
    public Collection<Flight> findFlightByDestinationDate(String destination, LocalDate date) throws Exception {
        DTODestinationDate destinationDate = new DTODestinationDate(destination, date);
        Request request = new Request.Builder().type(RequestType.GET_DD_FLIGHT).data(destinationDate).build();
        return getFlights(request);
    }

    @Override
    public Airport findAirportById(int id) throws Exception {
        Request request = new Request.Builder().type(RequestType.GET_AIRPORT).data(id).build();
        sendRequest(request);
        Response response = readResponse();
        if (response.type() == ResponseType.ERROR) {
            String err = response.data().toString();
            throw new Exception(err);
        }
        return (Airport) response.data();
    }

    @Override
    public Flight findFlightById(int id) throws Exception {
        Request request = new Request.Builder().type(RequestType.GET_FLIGHT).data(id).build();
        sendRequest(request);
        Response response = readResponse();
        if (response.type() == ResponseType.ERROR) {
            String err = response.data().toString();
            throw new Exception(err);
        }
        return (Flight) response.data();
    }

    @Override
    public Collection<Flight> getAllAvailableFlights() throws Exception {
        Request request = new Request.Builder().type(RequestType.GET_A_FLIGHTS).build();
        return getFlights(request);
    }

    @Override
    public Collection<Airport> getAllAirports() throws Exception {
        Request request = new Request.Builder().type(RequestType.GET_ALL_AIRPORTS).build();
        sendRequest(request);
        Response response = readResponse();
        if (response.type() == ResponseType.ERROR) {
            String err = response.data().toString();
            throw new Exception(err);
        }
        return (Collection<Airport>) response.data();
    }

    private Collection<Flight> getFlights(Request request) throws Exception {
        sendRequest(request);
        Response response = readResponse();
        if (response.type() == ResponseType.ERROR) {
            String err = response.data().toString();
            throw new Exception(err);
        }
        return (Collection<Flight>) response.data();
    }

//    @Override
//    public void updateFlight(int id, int numberOfSeats) throws Exception {
//        Request request = new Request.Builder().type(RequestType.UPDATE_FLIGHT).data().build();
//        sendRequest(request);
//        Response response = readResponse();
//        if (response.type() == ResponseType.ERROR) {
//            String err = response.data().toString();
//            throw new Exception(err);
//        }
//    }

    @Override
    public void buyTicket(Client client, List<Person> people, Flight flight) throws Exception {
        DTOFlight airportFlight = new DTOFlight(client, people, flight);
        Request request = new Request.Builder().type(RequestType.BUY_TICKET).data(airportFlight).build();
        sendRequest(request);
        Response response = readResponse();
        if (response.type() == ResponseType.ERROR) {
            String err = response.data().toString();
            throw new Exception(err);
        }
        //return (int) response.data();
    }

    private void sendRequest(Request request) throws Exception {
        try {
            output.writeObject(request);
            output.flush();
        } catch (IOException e) {
            throw new Exception("Error sending object " + e);
        }
    }

    private Response readResponse() throws Exception {
        Response response;
        try {
            response = qresponses.take();
        } catch (InterruptedException e) {
            throw new Exception("Error reading object " + e);
        }
        return response;
    }

    private void initializeConnection() {
        try {
            connection = new Socket(host, port);
            output = new ObjectOutputStream(connection.getOutputStream());
            output.flush();
            input = new ObjectInputStream(connection.getInputStream());
            finished = false;
            startReader();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void startReader() {
        Thread tw = new Thread(new ReaderThread());
        tw.start();
    }

    private void closeConnection() {
        finished = true;
        try {
            input.close();
            output.close();
            connection.close();
            client = null;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private boolean isUpdate(Response response) {
        return response.type() == ResponseType.TICKET_BOUGHT;
    }

    private void handleUpdate(Response response) throws Exception {
        if (response.type() == ResponseType.TICKET_BOUGHT) {
            System.out.println("Bought ticket update" + response.data());
            Collection<Flight> flights = (Collection<Flight>) response.data();
            client.ticketBought(flights);
        }
    }

    private class ReaderThread implements Runnable {
        public void run() {
            while (!finished) {
                try {
                    Object response = input.readObject();
                    System.out.println("response received " + response);
                    if (isUpdate((Response) response)) {
                        handleUpdate((Response) response);
                    } else {
                        try {
                            qresponses.put((Response) response);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                } catch (Exception e) {
                    System.out.println("Reading error " + e);
                }
            }
        }
    }
}
