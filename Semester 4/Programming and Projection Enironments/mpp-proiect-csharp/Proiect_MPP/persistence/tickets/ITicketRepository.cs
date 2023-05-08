using model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistence.tickets
{
    public interface ITicketRepository : Repository<Ticket, int>
    {
        Ticket? findByTouristName(string name);
        List<Ticket> getByInvoiceId(int invoiceId);
        List<Ticket> getByFlightId(int flightId);
    }
}
