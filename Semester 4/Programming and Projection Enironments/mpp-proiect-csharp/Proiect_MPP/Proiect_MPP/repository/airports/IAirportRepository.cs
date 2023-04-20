using Proiect_MPP.domain;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.airports
{
    internal interface IAirportRepository : Repository<Airport, int>
    {
        IList<Airport> getAirportAfterName(string name);
        IList<Airport> getAirportsInCity(string cityName);
    }
}
