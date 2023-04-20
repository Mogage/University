using Proiect_MPP.domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.tickets
{
    internal interface ITicketRepository : Repository<Ticket, int>
    {
        Ticket? findByTouristName(string name);
        IEnumerable<Ticket> getByInvoiceId(int invoiceId);
        IEnumerable<Ticket> getByFlightId(int flightId);
    }
}
