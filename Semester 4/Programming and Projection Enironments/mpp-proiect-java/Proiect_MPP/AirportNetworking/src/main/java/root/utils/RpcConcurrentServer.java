package root.utils;

import root.rpcprotocol.ClientRpcReflectionWorker;
import root.services.IService;

import java.net.Socket;


public class RpcConcurrentServer extends AbsConcurrentServer {
    private final IService chatServer;

    public RpcConcurrentServer(int port, IService chatServer) {
        super(port);
        this.chatServer = chatServer;
        System.out.println("Airport - RpcConcurrentServer");
    }

    @Override
    protected Thread createWorker(Socket client) {
        // ChatClientRpcWorker worker=new ChatClientRpcWorker(chatServer, client);
        ClientRpcReflectionWorker worker = new ClientRpcReflectionWorker(chatServer, client);

        return new Thread(worker);
    }

    @Override
    public void stop() {
        System.out.println("Stopping services ...");
    }
}
