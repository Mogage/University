using Proiect_MPP.domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.flights
{
    internal interface IFlightRepository : Repository<Flight, int>
    {
        IEnumerable<Flight> getByDepartureAirport(int departureAirport);
        IEnumerable<Flight> getByDestinationAirport(int destinationAirport);
        IEnumerable<Flight> getAfterDepartureDateTime(DateTime departureDateTime);
        IEnumerable<Flight> getAvailable();
    }
}
