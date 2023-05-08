using model;
using model.people;
using persistence.invoices;
using persistence.people.clients;
using persistence.tickets;
using services;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace client
{
    public partial class BuyTicketForm : Form
    {
        private Controller controller;
        private MainForm mainForm;
        private Flight Flight;

        public BuyTicketForm(Controller controller, Flight Flight, MainForm mainForm)
        {
            InitializeComponent();
            this.controller = controller;
            this.mainForm = mainForm;
            this.Flight = Flight;
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
                controller.Service.buyTicket(client, people, Flight);
                //int numberOfSeats = controller.Service.buyTicket(client, people, Flight);
                //mainForm.updateFlights(Flight.ID, Flight.FreeSeats - numberOfSeats);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }
    }
}
