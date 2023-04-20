package root.proiect_mpp.service.tickets;

import root.proiect_mpp.domain.DTOAirportFlight;
import root.proiect_mpp.domain.Invoice;
import root.proiect_mpp.domain.Ticket;
import root.proiect_mpp.domain.people.Client;
import root.proiect_mpp.domain.people.Person;
import root.proiect_mpp.repositories.invoices.InvoiceRepository;
import root.proiect_mpp.repositories.people.clients.ClientRepository;
import root.proiect_mpp.repositories.tickets.TicketRepository;

import java.util.List;

public class TicketService implements ITickerService {
    private final TicketRepository ticketRepository;
    private final InvoiceRepository invoiceRepository;
    private final ClientRepository clientRepository;

    public TicketService(TicketRepository ticketRepository, InvoiceRepository invoiceRepository, ClientRepository clientRepository) {
        this.ticketRepository = ticketRepository;
        this.invoiceRepository = invoiceRepository;
        this.clientRepository = clientRepository;
    }

    private void addTicket(int flightId, int invoiceId, String touristName) {
        ticketRepository.add(new Ticket(flightId, invoiceId, 1, touristName));
    }

    @Override
    public int buyTicket(Client client, List<Person> people, DTOAirportFlight airportFlight) throws Exception {
        if (client.getFirstName().isBlank() || client.getLastName().isBlank() || client.getAddress().isBlank()) {
            throw new Exception("Client can't be empty");
        }
        if (null == clientRepository.findByAddress(client.getAddress())) {
            clientRepository.add(client);
        }
        int numberOfTickets = 1;
        for (Person person : people) {
            if (person.getFirstName().isBlank() || person.getLastName().isBlank()) {
                break;
            }
            numberOfTickets = numberOfTickets + 1;
        }
        if (numberOfTickets > airportFlight.getFreeSeats()) {
            throw new Exception("There are not this many free seats.");
        }

        Invoice invoice = new Invoice(clientRepository.findByAddress(client.getAddress()).getId());
        invoice.setId(invoiceRepository.add(invoice));
        addTicket(airportFlight.getId(), invoice.getId(), client.getFirstName() + " " + client.getLastName());
        for (Person person : people) {
            if (person.getFirstName().isBlank() || person.getLastName().isBlank()) {
                break;
            }
            addTicket(airportFlight.getId(), invoice.getId(), person.getFirstName() + " " + person.getLastName());
        }

        return numberOfTickets;
    }
}
