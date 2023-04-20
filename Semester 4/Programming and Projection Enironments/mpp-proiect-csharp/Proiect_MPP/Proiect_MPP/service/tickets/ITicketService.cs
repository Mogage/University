using Proiect_MPP.domain;
using Proiect_MPP.domain.people;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.service.tickets
{
    internal interface ITicketService
    {
        int buyTicket(Client client, List<Person> people, Flight airportFlight);
    }
}
