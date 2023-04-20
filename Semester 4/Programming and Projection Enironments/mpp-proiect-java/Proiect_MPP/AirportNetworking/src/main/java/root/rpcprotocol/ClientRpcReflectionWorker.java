package root.rpcprotocol;

import root.model.Airport;
import root.model.DTOFlight;
import root.model.Flight;
import root.model.Ticket;
import root.model.DTODestinationDate;
import root.model.people.Employee;
import root.services.IObserver;
import root.services.IService;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.Socket;
import java.util.Collection;

public class ClientRpcReflectionWorker implements Runnable, IObserver {
    private final IService server;
    private final Socket connection;
    private ObjectInputStream input;
    private ObjectOutputStream output;
    private volatile boolean connected;

    public ClientRpcReflectionWorker(IService server, Socket connection) {
        this.server = server;
        this.connection = connection;
        try {
            output = new ObjectOutputStream(connection.getOutputStream());
            output.flush();
            input = new ObjectInputStream(connection.getInputStream());
            connected = true;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void run() {
        while (connected) {
            try {
                Object request = input.readObject();
                Response response = handleRequest((Request) request);
                if (response != null) {
                    sendResponse(response);
                }
            } catch (IOException | ClassNotFoundException e) {
                e.printStackTrace();
            }
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        try {
            input.close();
            output.close();
            connection.close();
        } catch (IOException e) {
            System.out.println("Error " + e);
        }
    }

    @Override
    public void ticketBought(Ticket ticket) throws Exception {
        try {
            sendResponse(new Response.Builder().type(ResponseType.TICKET_BOUGHT).data(ticket).build());
        } catch (IOException e) {
            throw new Exception("sending error: " + e);
        }
    }

    private static final Response okResponse = new Response.Builder().type(ResponseType.OK).build();

    //  private static Response errorResponse=new Response.Builder().type(ResponseType.ERROR).build();
    private Response handleRequest(Request request) {
        Response response = null;
        String handlerName = "handle" + (request).type();
        System.out.println("HandlerName " + handlerName);
        try {
            Method method = this.getClass().getDeclaredMethod(handlerName, Request.class);
            response = (Response) method.invoke(this, request);
            System.out.println("Method " + handlerName + " invoked");
        } catch (NoSuchMethodException | InvocationTargetException | IllegalAccessException e) {
            e.printStackTrace();
        }

        return response;
    }

    private Response handleLOGIN(Request request) {
        System.out.println("Login request ..." + request.type());
        System.out.println("Received request: " + request.data().toString());
        Employee employee = (Employee) request.data();
        try {
            Employee found = server.login(employee, this);
            return new Response.Builder().type(ResponseType.OK).data(found).build();
        } catch (Exception e) {
            connected = false;
            return new Response.Builder().type(ResponseType.ERROR).data(e.getMessage()).build();
        }
    }

    private Response handleLOGOUT(Request request) {
        System.out.println("Logout request...");
        Employee employee = (Employee) request.data();
        try {
            server.logout(employee, this);
            connected = false;
            return okResponse;

        } catch (Exception e) {
            return new Response.Builder().type(ResponseType.ERROR).data(e.getMessage()).build();
        }
    }

    private Response handleGET_A_FLIGHTS(Request request) {
        System.out.println("Get all flights request...");
        try {
            Collection<Flight> flights = server.getAllAvailableFlights();
            return new Response.Builder().type(ResponseType.GET_A_FLIGHTS).data(flights).build();
        } catch (Exception e) {
            return new Response.Builder().type(ResponseType.ERROR).data(e.getMessage()).build();
        }
    }

    private Response handleGET_FLIGHT(Request request) {
        System.out.println("Get flight request...");
        try {
            Flight flight = server.findFlightById((int) request.data());
            return new Response.Builder().type(ResponseType.GET_FLIGHT).data(flight).build();
        } catch (Exception e) {
            return new Response.Builder().type(ResponseType.ERROR).data(e.getMessage()).build();
        }
    }

    private Response handleUPDATE_FLIGHT(Request request) {
        System.out.println("Update flight request...");
        try {
            Flight flight = (Flight) request.data();
            server.updateFlight(flight, flight.getId());
            return okResponse;
        } catch (Exception e) {
            return new Response.Builder().type(ResponseType.ERROR).data(e.getMessage()).build();
        }
    }

    private Response handleGET_AIRPORT(Request request) {
        System.out.println("Get airport request...");
        try {
            Airport airport = server.findAirportById((int) request.data());
            return new Response.Builder().type(ResponseType.GET_AIRPORT).data(airport).build();
        } catch (Exception e) {
            return new Response.Builder().type(ResponseType.ERROR).data(e.getMessage()).build();
        }
    }

    private Response handleGET_DD_FLIGHT(Request request) {
        System.out.println("Get DD flight request...");
        try {
            DTODestinationDate dds = (DTODestinationDate) request.data();
            Collection<Flight> flights = server.findFlightByDestinationDate(dds.getDestination(), dds.getDate());
            return new Response.Builder().type(ResponseType.GET_DD_FLIGHT).data(flights).build();
        } catch (Exception e) {
            return new Response.Builder().type(ResponseType.ERROR).data(e.getMessage()).build();
        }
    }

    private Response handleBUY_TICKET(Request request){
        System.out.println("Buy ticket request...");
        try {
            DTOFlight dto = (DTOFlight) request.data();
            int numberOfSeats = server.buyTicket(dto.getClient(), dto.getPeople(), dto.getFlight());
            return new Response.Builder().type(ResponseType.OK).data(numberOfSeats).build();
        } catch (Exception e) {
            return new Response.Builder().type(ResponseType.ERROR).data(e.getMessage()).build();
        }
    }

    private void sendResponse(Response response) throws IOException {
        System.out.println("sending response " + response);
        output.writeObject(response);
        output.flush();
    }
}
