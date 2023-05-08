using log4net;
using model;
using model.people;
using network.utils;
using services;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace network
{
    public class ClientRpcProxy : IService
    {
        private static readonly ILog Logger = LogManager.GetLogger("ServicesRpcProxy");
        private readonly string host;
        private readonly int port;
        private IObserver employeeObserver;
        private NetworkStream stream;
        private IFormatter formatter;
        private TcpClient connection;
        private readonly Queue<Response> responses;
        private volatile bool finished;
        private EventWaitHandle waitHandle;

        public ClientRpcProxy(string host, int port)
        {
            this.host = host;
            this.port = port;
            responses = new Queue<Response>();
        }

        private void initializeConnection()
        {
            try
            {
                connection = new TcpClient(host, port);
                stream = connection.GetStream();
                formatter = new BinaryFormatter();
                finished = false;
                waitHandle = new AutoResetEvent(false);
                startReader();
            }
            catch (Exception e)
            {
                Console.WriteLine(e.StackTrace);
            }
        }

        private void closeConnection()
        {
            finished = true;
            try
            {
                stream.Close();
                connection.Close();
                waitHandle.Close();
                employeeObserver = null;
            }
            catch (Exception e)
            {
                Console.WriteLine(e.StackTrace);
            }
        }

        private void sendRequest(Request request)
        {
            try
            {
                formatter.Serialize(stream, request);
                stream.Flush();
            }
            catch (Exception e)
            {
                throw new ServerException("Error sending object ", e);
            }
        }

        private Response readResponse()
        {
            Response response = null;
            try
            {
                waitHandle.WaitOne();
                lock (responses)
                {
                    response = responses.Dequeue();
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(e.StackTrace);
            }
            return response;
        }

        private void startReader()
        {
            Thread tw = new Thread(run);
            tw.Start();
        }

        private static bool isUpdate(Response response)
        {
            return response.Type == ResponseType.TICKET_BOUGHT;
        }

        private void handleUpdate(Response response)
        {
            if (response.Type != ResponseType.TICKET_BOUGHT)
            {
                return;
            }

            var flights = (List<Flight>)response.Data;
            try
            {
                employeeObserver.ticketBought(flights);
            }
            catch (Exception e)
            {
                Logger.Error("Error handle update: " + e);
            }
        }

        public virtual void run()
        {
            while (!finished)
            {
                try
                {
                    object response = formatter.Deserialize(stream);
                    Console.WriteLine("response received " + response);
                    if (isUpdate((Response)response))
                    {
                        handleUpdate((Response)response);
                    }
                    else
                    {
                        lock (responses)
                        {
                            responses.Enqueue((Response)response);
                        }
                        waitHandle.Set();
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine("Reading error " + e);
                }
            }
        }

        public virtual Employee login(Employee employee, IObserver client)
        {
            initializeConnection();
            Employee employee1 = null;
            sendRequest(new Request.Builder().Type(RequestType.LOGIN).Data(employee).Build());
            Response response = readResponse();
            if (response.Type == ResponseType.OK)
            {
                employee1 = (Employee)response.Data;
                employeeObserver = client;
                return employee1;
            }
            if (response.Type == ResponseType.ERROR)
            {
                string err = response.Data.ToString();
                closeConnection();
                throw new ServerException(err);
            }
            return null;
        }

        public virtual void logout(Employee employee, IObserver client)
        {
            Request request = new Request.Builder().Type(RequestType.LOGOUT).Data(employee).Build();
            sendRequest(request);
            Response response = readResponse();
            closeConnection();
            if (response.Type == ResponseType.ERROR)
            {
                string err = response.Data.ToString();
                throw new ServerException(err);
            }
        }

        private List<Flight> getFlights(Request request)
        {
            sendRequest(request);
            Response response = readResponse();
            if (response.Type == ResponseType.ERROR)
            {
                string err = response.Data.ToString();
                throw new Exception(err);
            }
            return (List<Flight>)response.Data;
        }

        public virtual List<Flight> findFlightByDestinationDate(string destination, DateTime date)
        {
            DTODestinationDate destinationDate = new DTODestinationDate(destination, date);
            return getFlights(new Request.Builder().Type(RequestType.GET_DD_FLIGHT).Data(destinationDate).Build());
        }

        public virtual Airport findAirportById(int id)
        {
            Request request = new Request.Builder().Type(RequestType.GET_AIRPORT).Data(id).Build();
            sendRequest(request);
            Response response = readResponse();
            if (response.Type == ResponseType.ERROR)
            {
                String err = response.Data.ToString();
                throw new Exception(err);
            }
            return (Airport)response.Data;
        }

        public virtual Flight findFlightById(int id)
        {
            Request request = new Request.Builder().Type(RequestType.GET_FLIGHT).Data(id).Build();
            sendRequest(request);
            Response response = readResponse();
            if (response.Type == ResponseType.ERROR)
            {
                String err = response.Data.ToString();
                throw new Exception(err);
            }
            return (Flight)response.Data;
        }

        public virtual List<Flight> getAllAvailableFlights()
        {
            return getFlights(new Request.Builder().Type(RequestType.GET_A_FLIGHTS).Data(null).Build());
        }

        public virtual List<Airport> getAllAirports()
        {
            sendRequest(new Request.Builder().Type(RequestType.GET_ALL_AIRPORTS).Build());
            Response response = readResponse();
            if (response.Type == ResponseType.ERROR)
            {
                string err = response.Data.ToString();
                throw new ServerException(err);
            }
            return (List<Airport>)response.Data;
        }

        public virtual void buyTicket(Client client, List<Person> people, Flight flight)
        {
            DTOFlight airportFlight = new DTOFlight(client, people, flight);
            sendRequest(new Request.Builder().Type(RequestType.BUY_TICKET).Data(airportFlight).Build());
            Response response = readResponse();

            if (response.Type == ResponseType.ERROR)
            {
                string err = response.Data.ToString();
                throw new ServerException(err);
            }
        }
    }
}
