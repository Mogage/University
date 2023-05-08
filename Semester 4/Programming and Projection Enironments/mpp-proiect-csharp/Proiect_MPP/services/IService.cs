using model.people;
using model;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace services
{
    public interface IService
    {
        Employee login(Employee employee, IObserver client);
        void logout(Employee employee, IObserver client);

        List<Flight> findFlightByDestinationDate(string destination, DateTime date);
        Airport findAirportById(int id);
        Flight findFlightById(int id);
        List<Flight> getAllAvailableFlights();
        List<Airport> getAllAirports();

        void buyTicket(Client client, List<Person> people, Flight flight);
    }
}
