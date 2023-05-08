using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace network.utils
{
    public abstract class AbstractServer
    {
        private TcpListener server;
        private string host;
        private int port;

        public AbstractServer(string host, int port)
        {
            this.host = host;
            this.port = port;
        }

        public void Start()
        {
            IPAddress adr = IPAddress.Parse(host);
            IPEndPoint ep = new IPEndPoint(adr, port);
            server = new TcpListener(ep);
            server.Start();
            while (true)
            {
                Console.WriteLine("Waiting for clients ...");
                TcpClient client = server.AcceptTcpClient();
                Console.WriteLine("Client connected ...");
                ProcessRequest(client);
            }
        }

        public abstract void ProcessRequest(TcpClient client);

        public void Stop()
        {
            server.Stop();
            Console.WriteLine("Server closed.");
        }
    }
}
