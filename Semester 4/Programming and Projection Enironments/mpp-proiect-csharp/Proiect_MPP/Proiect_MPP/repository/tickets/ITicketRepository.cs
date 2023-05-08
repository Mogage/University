using Proiect_MPP.domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.tickets
{
    public interface ITicketRepository : Repository<Ticket, int>
    {
        Ticket? findByTouristName(string name);
        List<Ticket> getByInvoiceId(int invoiceId);
        List<Ticket> getByFlightId(int flightId);
    }
}
