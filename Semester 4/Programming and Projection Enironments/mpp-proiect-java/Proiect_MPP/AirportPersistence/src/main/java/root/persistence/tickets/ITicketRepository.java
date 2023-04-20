package root.persistence.tickets;

import root.model.Ticket;
import root.persistence.Repository;

import java.util.Collection;

public interface ITicketRepository extends Repository<Ticket, Integer> {
    Ticket findByTouristName(String touristName);
    Collection<Ticket> getByInvoiceId(int invoiceID);
    Collection<Ticket> getByFlightId(int flightID);
}
