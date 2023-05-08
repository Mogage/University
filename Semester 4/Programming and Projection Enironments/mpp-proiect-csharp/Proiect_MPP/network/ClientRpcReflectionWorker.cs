using services;
using model;
using model.people;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using System.Reflection;
using System.Collections.ObjectModel;

namespace network
{
    public class ClientRpcReflectionWorker : IObserver
    {
        private IService server;
        private TcpClient connection;

        private readonly NetworkStream stream;
        private readonly IFormatter formatter;
        private volatile bool connected;

        public ClientRpcReflectionWorker(IService server, TcpClient connection)
        {
            this.server = server;
            this.connection = connection;
            try
            {
                stream = connection.GetStream();
                formatter = new BinaryFormatter();
                connected = true;
            }
            catch (Exception e)
            {
                Console.WriteLine(e.StackTrace);
            }
        }

        public void ticketBought(List<Flight> flights)
        {
            try
            {
                sendResponse(new Response.Builder().Type(ResponseType.TICKET_BOUGHT).Data(flights).Build());
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.StackTrace);
            }
        }

        public virtual void Run()
        {
            while (connected)
            {
                try
                {
                    object request = formatter.Deserialize(stream);
                    object response = handleRequest((Request)request);
                    if (response != null)
                    {
                        sendResponse((Response)response);
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.StackTrace);
                }

                try
                {
                    Thread.Sleep(1000);
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.StackTrace);
                }
            }
            try
            {
                stream.Close();
                connection.Close();
            }
            catch (Exception e)
            {
                Console.WriteLine("Error " + e);
            }
        }

        private void sendResponse(Response response)
        {
            Console.WriteLine("sending response " + response);
            lock (stream)
            {
                formatter.Serialize(stream, response);
                stream.Flush();
            }
        }

        private Response handleRequest(Request request)
        {
            Response response = null;
            string handlerName = "handle" + request.Type;
            Console.WriteLine($"Handler name: {handlerName}");
            try
            {
                var method = GetType().GetMethod(handlerName, new[] { typeof(Request) });
                response = (Response)method.Invoke(this, new[] { request });
                Console.WriteLine("Method " + handlerName + " invoked");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.StackTrace);
            }
            return response;
        }

        private static readonly Response okResponse = new Response.Builder().Type(ResponseType.OK).Build();

        public Response handleLOGIN(Request request)
        {
            Console.WriteLine("Login request ..." + request.Type);
            Console.WriteLine("Received request: " + request.Data.ToString());
            Employee employee = (Employee)request.Data;
            try
            {
                Employee found = server.login(employee, this);
                return new Response.Builder().Type(ResponseType.OK).Data(found).Build();
            }
            catch (Exception ex)
            {
                connected = false;
                return new Response.Builder().Type(ResponseType.ERROR).Data(ex.StackTrace).Build();
            }
        }

        public Response handleGET_FLIGHT(Request request)
        {
            Console.WriteLine("Get flight request...");
            try
            {
                Flight flight = server.findFlightById((int)request.Data);
                return new Response.Builder().Type(ResponseType.GET_FLIGHT).Data(flight).Build();
            }
            catch (Exception e)
            {
                return new Response.Builder().Type(ResponseType.ERROR).Data(e.Message).Build();
            }
        }

        public Response handleLOGOUT(Request request)
        {
            Console.WriteLine("Logout request...");
            Employee employee = (Employee)request.Data;
            try
            {
                server.logout(employee, this);
                connected = false;
                return okResponse;
            }
            catch (Exception ex)
            {
                return new Response.Builder().Type(ResponseType.ERROR).Data(ex.StackTrace).Build();
            }
        }

        public Response handleGET_A_FLIGHTS(Request request)
        {
            Console.WriteLine("Get all flights request...");
            try
            {
                List<Flight> flights = server.getAllAvailableFlights().ToList();
                return new Response.Builder().Type(ResponseType.GET_A_FLIGHTS).Data(flights).Build();
            }
            catch (Exception e)
            {
                return new Response.Builder().Type(ResponseType.ERROR).Data(e.StackTrace).Build();
            }
        }

        public Response handleGET_ALL_AIRPORTS(Request request)
        {
            Console.WriteLine("Get airports request...");
            try
            {
                List<Airport> airports = server.getAllAirports();
                return new Response.Builder().Type(ResponseType.GET_ALL_AIRPORTS).Data(airports).Build();
            }
            catch (Exception ex)
            {
                return new Response.Builder().Type(ResponseType.ERROR).Data(ex.StackTrace).Build();
            }
        }

        public Response handleGET_DD_FLIGHT(Request request)
        {
            Console.WriteLine("Get DD flight request...");
            try
            {
                DTODestinationDate dds = (DTODestinationDate)request.Data;
                List<Flight> flights = server.findFlightByDestinationDate(dds.Destination, dds.Date);
                return new Response.Builder().Type(ResponseType.GET_DD_FLIGHT).Data(flights).Build();
            }
            catch (Exception ex)
            {
                return new Response.Builder().Type(ResponseType.ERROR).Data(ex.StackTrace).Build();
            }
        }

        public Response handleBUY_TICKET(Request request)
        {
            Console.WriteLine("Buy ticket request...");
            try
            {
                DTOFlight dto = (DTOFlight)request.Data;
                server.buyTicket(dto.client, dto.people, dto.flight);
                return okResponse;
            }
            catch (Exception ex)
            {
                return new Response.Builder().Type(ResponseType.ERROR).Data(ex.StackTrace).Build();
            }
        }
    }
}
