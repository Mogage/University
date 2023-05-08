using Proiect_MPP.domain;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.airports
{
    public interface IAirportRepository : Repository<Airport, int>
    {
        List<Airport> getAirportAfterName(string name);
        List<Airport> getAirportsInCity(string cityName);
    }
}
