using Proiect_MPP.domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.service.main
{
    public interface IMainService
    {
        void updateFlight(Flight flight, int id);

        List<Flight> findByDestinationDate(string destination, DateTime date);

        Airport? findAirportById(int id);

        Flight? findFlightById(int id);

        List<Flight> getAllAvailableFlights();
    }
}
