using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Net.Sockets;
using services;
using network.utils;
using log4net;

namespace Protobuff
{
    public class ConcurrentServerProto : AbsConcurrentServer
    {
        private readonly IService service;
        private static readonly ILog logger = LogManager.GetLogger("RpcConcurrentServer");

        private ProtoWorker worker;
        public ConcurrentServerProto(string host, int port, IService service) : base(host, port)
        {
            this.service = service;
            logger.Info("RpcConcurrentServer created");
            Console.WriteLine(@"RpcConcurrentServer created");
        }
        protected override Thread CreateWorker(TcpClient client)
        {
            worker = new ProtoWorker(service, client);
            return new Thread(new ThreadStart(worker.Run));
        }
    }
}
