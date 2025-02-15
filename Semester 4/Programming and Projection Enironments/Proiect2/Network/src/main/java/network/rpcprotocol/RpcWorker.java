package network.rpcprotocol;

import model.DtoInitialise;
import model.Game;
import model.Player;
import services.IObserver;
import services.IService;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.Socket;
import java.util.Collection;

public class RpcWorker implements Runnable, IObserver {
    private final IService service;
    private final Socket connection;
    private ObjectInputStream input;
    private ObjectOutputStream output;
    private volatile boolean connected;

    public RpcWorker(IService service, Socket connection) {
        this.service = service;
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
            e.printStackTrace();
        }
    }

    private void sendResponse(Response response) throws IOException {
        System.out.println("sending response " + response);
        output.writeObject(response);
        output.flush();
    }

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
        Player player = (Player) request.data();
        try {
            Player found = service.login(player, this);
            return new Response.Builder().type(ResponseType.OK).data(found).build();
        } catch (Exception e) {
            connected = false;
            return new Response.Builder().type(ResponseType.ERROR).data(e.getMessage()).build();
        }
    }

    private Response handleLOGOUT(Request request) {
        System.out.println("Logout request...");
        Player player = (Player) request.data();
        try {
            service.logout(player);
            connected = false;
            return new Response.Builder().type(ResponseType.OK).build();
        } catch (Exception e) {
            return new Response.Builder().type(ResponseType.ERROR).data(e.getMessage()).build();
        }
    }

    private Response handleINITIALISE(Request request) {
        System.out.println("Initialise request ..." + request.type());
        System.out.println("Received request: " + request.data().toString());
        int id = (int) request.data();
        try {
            DtoInitialise found = service.initialise(id);
            return new Response.Builder().type(ResponseType.OK).data(found).build();
        } catch (Exception e) {
            connected = false;
            return new Response.Builder().type(ResponseType.ERROR).data(e.getMessage()).build();
        }
    }

    private Response handleMOVE(Request request) {
        System.out.println("Move request ..." + request.type());
        System.out.println("Received request: " + request.data().toString());
        int id = (int) request.data();
        try {
            int found = service.move(id);
            return new Response.Builder().type(ResponseType.OK).data(found).build();
        } catch (Exception e) {
            connected = false;
            return new Response.Builder().type(ResponseType.ERROR).data(e.getMessage()).build();
        }
    }

    private Response handleGET_SCORE(Request request) {
        System.out.println("Get score request ..." + request.type());
        System.out.println("Received request: " + request.data().toString());
        int id = (int) request.data();
        try {
            int found = service.getScore(id);
            return new Response.Builder().type(ResponseType.OK).data(found).build();
        } catch (Exception e) {
            connected = false;
            return new Response.Builder().type(ResponseType.ERROR).data(e.getMessage()).build();
        }
    }

    @Override
    public void gameFinished(Collection<Game> gamesList) throws IOException {
        sendResponse(new Response.Builder().type(ResponseType.GAME_FINISHED).data(gamesList).build());
    }
}
