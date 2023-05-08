using Proiect_MPP.domain;
using Proiect_MPP.domain.people;
using Proiect_MPP.repository.invoices;
using Proiect_MPP.repository.people.clients;
using Proiect_MPP.repository.tickets;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.service.tickets
{
    public class TicketService : ITicketService
    {
        private readonly ITicketRepository ticketRepository;
        private readonly IInvoiceRepository invoiceRepository;
        private readonly IClientRepository clientRepository;

        public TicketService(ITicketRepository ticketRepository, IInvoiceRepository invoiceRepository, IClientRepository clientRepository)
        {
            this.ticketRepository = ticketRepository;
            this.invoiceRepository = invoiceRepository;
            this.clientRepository = clientRepository;
        }

        private void addTicket(int flightId, int invoiceId, string touristName)
        {
            ticketRepository.add(new Ticket(flightId, invoiceId, 1, touristName));

        }

        public int buyTicket(Client client, List<Person> people, Flight airportFlight)
        {
            if (string.IsNullOrWhiteSpace(client.FirstName) || string.IsNullOrWhiteSpace(client.LastName) || string.IsNullOrWhiteSpace(client.Address))
            {
                throw new Exception("Client can't be empty.");
            }
            if (null == clientRepository.findByAddress(client.Address))
            {
                clientRepository.add(client);
            }
            int numberOfTickets = 1;
            foreach (Person person in people) 
            {
                if(string.IsNullOrWhiteSpace(person.FirstName) || string.IsNullOrWhiteSpace(person.LastName))
                {
                    break;
                }
                numberOfTickets = numberOfTickets + 1;
            }
            
            if (numberOfTickets > airportFlight.FreeSeats)
            {
                throw new Exception("There are not that many free seats.");
            }

            Invoice invoice = new Invoice(clientRepository.findByAddress(client.Address).ID);
            invoice.ID = invoiceRepository.add(invoice);
            addTicket(airportFlight.ID, invoice.ID, client.FirstName + " " + client.LastName);
            foreach (Person person in people)
            {
                if (string.IsNullOrWhiteSpace(person.FirstName) || string.IsNullOrWhiteSpace(person.LastName))
                {
                    break;
                }
                addTicket(airportFlight.ID, invoice.ID, person.FirstName + " " + person.LastName);
            }

            return numberOfTickets;
        }
    }
}
