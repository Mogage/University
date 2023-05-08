using Server;
using persistence.airports;
using persistence.flights;
using persistence.invoices;
using persistence.tickets;
using persistence.people.clients;
using persistence.people.employees;
using services;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using network.utils;
using services;
using server;
using Protobuff;

namespace Server
{
    internal static class StartRPCServer
    {
        
        private static string GetConnectionStringByName(string name)
        {
            string returnValue = null;
            var settings = ConfigurationManager.ConnectionStrings[name];
            if (settings != null)
                returnValue = settings.ConnectionString;
            return returnValue;
        }

        [STAThread]
        public static void Main(string[] args)
        {

            IDictionary<string, string> props = new SortedList<string, string>();
            props.Add("ConnectionString", GetConnectionStringByName("tourismDB"));
            IEmployeeRepository employeeRepository = new EmployeeRepository(props);
            ITicketRepository ticketRepository = new TicketRepository(props);
            IFlightRepository flightRepository = new FlightRepository(props);
            IAirportRepository airportRepository = new AirportRepository(props);
            IInvoiceRepository invoiceRepository = new InvoiceRepository(props);
            IClientRepository clientRepository = new ClientRepository(props);

            IService service = new ServiceImpl(employeeRepository, flightRepository, invoiceRepository, ticketRepository,
               airportRepository, clientRepository);


            // var server = new RpcConcurrentServer("127.0.0.1", 55558, service);
            var server = new ConcurrentServerProto("127.0.0.1", 55558, service);
            try
            {
                server.Start();
                Console.WriteLine(@"Server started ...");
                Console.ReadLine();
            }
            catch (ServerException e)
            {
                Console.Error.WriteLine("Error starting the server" + e.Message);
            }
            finally
            {
                try
                {
                    server.Stop();
                }
                catch (ServerException e)
                {
                    Console.Error.WriteLine("Error stopping server " + e.Message);
                }
            }

        }
    }
}