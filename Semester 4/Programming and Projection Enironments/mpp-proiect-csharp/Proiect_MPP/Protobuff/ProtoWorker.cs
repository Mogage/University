using services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.Serialization.Formatters.Binary;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using Google.Protobuf;
using ProiectMpp.Protocol;
using Flight = model.Flight;
using Employee = model.people.Employee;
using DTODestinationDate = model.DTODestinationDate;
using DTOFlight = model.DTOFlight;

namespace Protobuff
{
    public class ProtoWorker : IObserver
    {
        private IService server;
        private TcpClient connection;

        private readonly NetworkStream stream;
        private volatile bool connected;

        public ProtoWorker(IService server, TcpClient connection)
        {
            this.server = server;
            this.connection = connection;
            try
            {
                stream = connection.GetStream();
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
                sendResponse(Utils.CreateTicketBoughtResponse(flights));
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
                    object request = Request.Parser.ParseDelimitedFrom(stream);
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
                response.WriteDelimitedTo(stream);
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

        private static readonly Response okResponse = new Response { Type = Response.Types.Type.Ok };

        public Response handleLogin(Request request)
        {
            Console.WriteLine("Login request ..." + request.Type);
            Console.WriteLine("Received request: " + request.ToString());
            Employee employee = Utils.GetEmployee(request);
            try
            {
                return Utils.CreateLoginResponse(server.login(employee, this));
            }
            catch (Exception ex)
            {
                connected = false;
                return Utils.CreateErrorResponse(ex.Message);
            }
        }

        public Response handleGetFlight(Request request)
        {
            Console.WriteLine("Get flight request...");
            try
            {
                return Utils.CreateGetFlightResponse(server.findFlightById(request.Id));
            }
            catch (Exception e)
            {
                return Utils.CreateErrorResponse(e.Message);
            }
        }

        public Response handleLogout(Request request)
        {
            Console.WriteLine("Logout request...");
            Employee employee = Utils.GetEmployee(request);
            try
            {
                lock(server)
                {
                    server.logout(employee, this);
                    connected = false;
                    return okResponse;
                }
            }
            catch (Exception ex)
            {
                return Utils.CreateErrorResponse(ex.Message);
            }
        }

        public Response handleGetAFlights(Request request)
        {
            Console.WriteLine("Get all flights request...");
            try
            {
                lock (server)
                {
                    return Utils.CreateGetAFlightsResponse(server.getAllAvailableFlights().ToList());
                }
            }
            catch (Exception e)
            {
                return Utils.CreateErrorResponse(e.Message);
            }
        }

        public Response handleGetAllAirports(Request request)
        {
            Console.WriteLine("Get airports request...");
            try
            {
                lock (server)
                {
                    return Utils.CreateGetAllAirportsResponse(server.getAllAirports());
                }
            }
            catch (Exception ex)
            {
                return Utils.CreateErrorResponse(ex.Message);
            }
        }

        public Response handleGetDdFlight(Request request)
        {
            Console.WriteLine("Get DD flight request...");
            try
            {
                lock (server)
                {
                    DTODestinationDate dds = Utils.GetDtoDestinationDate(request);
                    Console.WriteLine(dds.ToString());
                    List<Flight> flights = server.findFlightByDestinationDate(dds.Destination, dds.Date);
                    return Utils.CreateGetDDFlights(flights);
                }
            }
            catch (Exception ex)
            {
                return Utils.CreateErrorResponse(ex.Message);
            }
        }

        public Response handleBuyTicket(Request request)
        {
            Console.WriteLine("Buy ticket request...");
            try
            {
                lock (server)
                {
                    DTOFlight dto = Utils.GetFlightDetails(request);
                    server.buyTicket(dto.client, dto.people, dto.flight);
                    return okResponse;
                }
            }
            catch (Exception ex)
            {
                return Utils.CreateErrorResponse(ex.Message);
            }
        }
    }
}
