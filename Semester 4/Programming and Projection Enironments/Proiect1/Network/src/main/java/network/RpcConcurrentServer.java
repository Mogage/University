package network;

import network.rpcprotocol.RpcReflectionWorker;
import network.utils.AbstractConcurentServer;
import services.IService;

import java.net.Socket;

public class RpcConcurrentServer extends AbstractConcurentServer {
    private final IService service;

    public RpcConcurrentServer(int port, IService chatServer) {
        super(port);
        this.service = chatServer;
        System.out.println("RpcConcurrentServer");
    }

    @Override
    protected Thread createWorker(Socket client) {
        RpcReflectionWorker worker = new RpcReflectionWorker(service, client);

        return new Thread(worker);
    }

    @Override
    public void stop() {
        System.out.println("Stopping services ...");
    }
}
