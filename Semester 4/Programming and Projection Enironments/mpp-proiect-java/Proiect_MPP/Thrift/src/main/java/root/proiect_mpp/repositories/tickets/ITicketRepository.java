package root.proiect_mpp.repositories.tickets;

import root.proiect_mpp.domain.Ticket;
import root.proiect_mpp.repositories.Repository;

import java.util.Collection;

public interface ITicketRepository extends Repository<Ticket, Integer> {
    Ticket findByTouristName(String touristName);
    Collection<Ticket> getByInvoiceId(int invoiceID);
    Collection<Ticket> getByFlightId(int flightID);
}
