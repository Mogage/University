using Proiect_MPP.domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.service.main
{
    internal interface IMainService
    {
        void updateFlight(Flight flight, int id);

        IEnumerable<Flight> findByDestinationDate(string destination, DateOnly date);

        Airport? findAirportById(int id);

        Flight? findFlightById(int id);

        IEnumerable<Flight> getAllAvailableFlights();
    }
}
