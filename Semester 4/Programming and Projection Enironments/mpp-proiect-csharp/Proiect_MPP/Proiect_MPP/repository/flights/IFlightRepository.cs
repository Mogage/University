﻿using Proiect_MPP.domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.flights
{
    public interface IFlightRepository : Repository<Flight, int>
    {
        List<Flight> getByDepartureAirport(int departureAirport);
        List<Flight> getByDestinationAirport(int destinationAirport);
        List<Flight> getAfterDepartureDateTime(DateTime departureDateTime);
        List<Flight> getAvailable();
    }
}
