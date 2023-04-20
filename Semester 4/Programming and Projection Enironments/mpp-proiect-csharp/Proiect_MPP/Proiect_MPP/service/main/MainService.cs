using Proiect_MPP.domain;
using Proiect_MPP.repository.flights;
using Proiect_MPP.repository.airports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.service.main
{
    internal class MainService : IMainService
    {
        private readonly FlightRepository flightRepository;
        private readonly AirportRepository airportRepository;

        public MainService(FlightRepository flightRepository, AirportRepository airportRepository)
        {
            this.flightRepository = flightRepository;
            this.airportRepository = airportRepository;
        }

        public void updateFlight(Flight flight, int id)
        {
            flightRepository.update(flight, id);
        }

        public Airport? findAirportById(int id)
        {
            return airportRepository.findById(id);
        }

        public Flight? findFlightById(int id)
        {
            return flightRepository.findById(id);
        }

        public IEnumerable<Flight> getAllAvailableFlights()
        {
            return flightRepository.getAvailable();
        }

        public IEnumerable<Flight> findByDestinationDate(string destination, DateOnly Date)
        {
            IEnumerable<Airport> airports = airportRepository.getAirportsInCity(destination);
            List<Flight> flightList = new List<Flight>();
            IEnumerable<Flight> flights;
            DateOnly departureDate;
            foreach (Airport airport in airports) 
            {
                flights = flightRepository.getByDestinationAirport(airport.ID);
                if (null == flights)
                {
                    continue;
                }

                foreach (Flight flight in flights)
                {
                    departureDate = flight.DepartureDate;
                    if (departureDate.Year != Date.Year ||
                        departureDate.Month != Date.Month || 
                        departureDate.Day != Date.Day ||
                        flight.FreeSeats == 0)
                    {
                        continue;
                    }
                    flightList.Add(flight);
                }
            }
            return flightList;
        }

    }
}
