using Proiect_MPP.domain;
using Proiect_MPP.domain.people;
using Proiect_MPP.repository.invoices;
using Proiect_MPP.repository.people.clients;
using Proiect_MPP.repository.tickets;
using Proiect_MPP.service.tickets;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Proiect_MPP
{
    public partial class BuyTicketForm : Form
    {
        ITicketService ticketService;
        IDictionary<string, string> properties;
        MainForm mainForm;
        Flight Flight;

        public BuyTicketForm(IDictionary<string, string> properties, Flight Flight, MainForm mainForm)
        {
            InitializeComponent();
            this.properties = properties;
            this.mainForm = mainForm;
            this.Flight = Flight;
            ticketService = new TicketService(new TicketRepository(properties), new InvoiceRepository(properties), new ClientRepository(properties));
        }

        private void BuyTicketForm_Load(object sender, EventArgs e)
        {

        }

        private void buyTicketButton_Click(object sender, EventArgs e)
        {
            Client client = new Client(clientFirstNameTextBox.Text, clientLastNameTextBox.Text, clientAddressTextBox.Text);
            List<Person> people = new List<Person>();
            if (!string.IsNullOrWhiteSpace(touristsTextBox.Text))
            {
                touristsTextBox.Text.Split(',').ToList().ForEach(tourist => { string[] names = tourist.Split(' '); people.Add(new Person(names[0], names[1])); });
            }
            
            try
            {
                int numberOfSeats = ticketService.buyTicket(client, people, Flight);
                mainForm.updateFlights(Flight.ID, Flight.FreeSeats - numberOfSeats);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }
    }
}
