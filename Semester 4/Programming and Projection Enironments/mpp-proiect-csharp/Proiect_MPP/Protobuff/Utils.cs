using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ProiectMpp.Protocol;
using Flight = model.Flight;
using Employee = model.people.Employee;
using Airport = model.Airport;
using DTODestinationDate = model.DTODestinationDate;
using DTOFlight = model.DTOFlight;
using Client = model.people.Client;
using Person = model.people.Person;
using System.Globalization;

namespace Protobuff
{
    public class Utils
    {
        public static Employee GetEmployee(Request request)
        {
            return new Employee(request.Employee.Id, request.Employee.FirstName, request.Employee.LastName,
                request.Employee.Position, request.Employee.Username, request.Employee.Password);
        }

        public static DTODestinationDate GetDtoDestinationDate(Request request) 
        {
            return new DTODestinationDate(request.DtoDestinationDate.Destination, DateTime.ParseExact(request.DtoDestinationDate.Date, "dd/MM/yyyy", CultureInfo.InvariantCulture));
        }

        public static DTOFlight GetFlightDetails(Request request)
        {
            Client myClient = new Client(request.DtoFlight.Client.Id, request.DtoFlight.Client.FirstName,
                request.DtoFlight.Client.LastName, request.DtoFlight.Client.Address);
            Flight myFlight = new Flight(request.DtoFlight.Flight.Id, request.DtoFlight.Flight.FreeSeats, request.DtoFlight.Flight.DestinationAirport,
                request.DtoFlight.Flight.DepartureAirport, DateTime.ParseExact(request.DtoFlight.Flight.DepartureDate, "dd/MM/yyyy", CultureInfo.InvariantCulture),
                DateTime.ParseExact(request.DtoFlight.Flight.DepartureTime, "mm:HH", CultureInfo.InvariantCulture));
            List<Person> myPersons = new List<Person>();
            foreach (ProiectMpp.Protocol.Person person in request.DtoFlight.Persons)
            {
                myPersons.Add(new Person(person.Id, person.FirstName, person.LastName));
            }
            return new DTOFlight(myClient, myPersons, myFlight);
        }

        public static Response CreateLoginResponse(Employee employee)
        {
            var responseEmployee = new ProiectMpp.Protocol.Employee()
            {
                Id = employee.ID,
                FirstName = employee.FirstName,
                LastName = employee.LastName,
                Position = employee.Position,
                Username = employee.Email,
                Password = employee.Password
            };
            return new Response { Type = Response.Types.Type.Ok, Employee = responseEmployee };
        }

        public static Response CreateErrorResponse(string message)
        {
            return new Response { Type = Response.Types.Type.Error, Error = message };
        }

        public static Response CreateGetFlightResponse(Flight flight)
        {
            var responseFlight = new ProiectMpp.Protocol.Flight()
            {
                Id = flight.ID,
                FreeSeats = flight.FreeSeats,
                DepartureAirport = flight.DepartureAirport,
                DestinationAirport = flight.DestinationAirport,
                DepartureDate = flight.DepartureDate.ToString("dd/MM/yyyy"),
                DepartureTime = flight.DepartureTime.ToString("hh:MM")
            };
            return new Response { Type = Response.Types.Type.GetFlight, Flight = responseFlight };
        }

        private static List<ProiectMpp.Protocol.Flight> CreateProtocolFlights(List<Flight> flights)
        {
            List<ProiectMpp.Protocol.Flight> responseFlights = new List<ProiectMpp.Protocol.Flight>();
            foreach (Flight flight in flights)
            {
                responseFlights.Add(new ProiectMpp.Protocol.Flight()
                {
                    Id = flight.ID,
                    DepartureAirport = flight.DepartureAirport,
                    DestinationAirport = flight.DestinationAirport,
                    FreeSeats = flight.FreeSeats,
                    DepartureDate = flight.DepartureDate.ToString("dd/MM/yyyy"),
                    DepartureTime = flight.DepartureTime.ToString("hh:MM")
                });
            }
            return responseFlights;
        }

        public static Response CreateGetAFlightsResponse(List<Flight> flights)
        {
            return new Response { Type = Response.Types.Type.GetAFlights, Flights = { CreateProtocolFlights(flights) } };
        }

        public static Response CreateGetAllAirportsResponse(List<Airport> airports)
        {
            List<ProiectMpp.Protocol.Airport> responseAirports = new List<ProiectMpp.Protocol.Airport>();
            foreach (Airport airport in airports)
            {
                responseAirports.Add(new ProiectMpp.Protocol.Airport()
                {
                    Id = airport.ID,
                    CityName = airport.CityName,
                    Name = airport.Name
                });
            }
            return new Response { Type = Response.Types.Type.GetAllAirports, Airports = { responseAirports } };
        }

        public static Response CreateGetDDFlights(List<Flight> flights)
        {
            return new Response { Type = Response.Types.Type.GetDdFlight, Flights = { CreateProtocolFlights(flights) } };
        }

        public static Response CreateTicketBoughtResponse(List<Flight> flights)
        {
            return new Response { Type = Response.Types.Type.TicketBought, Flights = { CreateProtocolFlights(flights) } };
        }

    }
}
