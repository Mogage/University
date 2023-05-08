using model;
using persistence.airports;
using persistence.flights;
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
using static log4net.Appender.RollingFileAppender;

namespace client
{
    public partial class MainForm : Form
    {
        private Controller controller;
        private BuyTicketForm buyTicketForm;
        private LogInForm logInForm;
        private List<Airport> airports = null;

        public MainForm(Controller controller, LogInForm logInForm)
        {
            InitializeComponent();
            this.controller = controller;
            this.logInForm = logInForm;
            controller.UpdateEvent += EmployeeUpdate;
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            List<Flight> flights = controller.Service.findFlightByDestinationDate("", DateTime.Now);
            List<Flight> availableFlights = controller.Service.getAllAvailableFlights();
            airports = controller.Service.getAllAirports();

            updateFlightsTable(availableFlights);
            updateSearchTable(flights);
            buyTicketsButton.Enabled = false;
        }

        private List<DTOAirportFlight> updateLists(List<Flight> flights)
        {
            List<DTOAirportFlight> dTOAirportFlights = new List<DTOAirportFlight>();
            Airport departure;
            Airport destination;

            foreach (Flight flight in flights)
            {
                departure = airports.Find(airport => airport.ID == flight.DepartureAirport);
                destination = airports.Find(airport => airport.ID == flight.DestinationAirport);
                dTOAirportFlights.Add(new DTOAirportFlight(flight.ID, departure.CityName, departure.Name, destination.CityName,
                    destination.Name, flight.DepartureDate, flight.DepartureTime, flight.FreeSeats));
            }


            return dTOAirportFlights;
        }

        private void updateTable(DataGridView table, List<DTOAirportFlight> dtoAirportFlights)
        {
            table.Rows.Clear();
            foreach (DTOAirportFlight dtoAirportFlight in dtoAirportFlights)
            {
                string[] row = { dtoAirportFlight.ID.ToString(), dtoAirportFlight.DepartureCity, dtoAirportFlight.DepartureName,
                    dtoAirportFlight.DestinationCity, dtoAirportFlight.DestinationName, dtoAirportFlight.DepartureDate.ToString("dd/MM/yyyy"),
                    dtoAirportFlight.DepartureTime.ToString("HH:mm"), dtoAirportFlight.FreeSeats.ToString() };

                table.Rows.Add(row);
            }
        }

        private void updateFlightsTable(List<Flight> flights)
        {
            List<DTOAirportFlight> dtoAirportFlights = updateLists(flights);
            updateTable(allFlightsTable, dtoAirportFlights);
        }

        private void updateSearchTable(List<Flight> flights)
        {
            string destination = destinationTextBox.Text;
            DateTime dateTime = datePicker.Value;
            DateTime date = new DateTime(dateTime.Year, dateTime.Month, dateTime.Day);
            List<DTOAirportFlight> dtoAirportFlights = updateLists(flights);
            updateTable(searchFlightsTable, dtoAirportFlights);
        }

        //public void updateFlights(int flightId, int numberOfSeats)
        //{
        //    Flight flight = controller.Service.findFlightById(flightId);
        //    flight.FreeSeats = numberOfSeats;
        //    controller.Service.updateFlight(flight, flightId);
        //    updateFlightsTable();
        //    updateSearchTable();
        //}

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);
            if (e.CloseReason == CloseReason.UserClosing)
            {
                controller.logout();
                //Application.Exit();
            }

            logInForm.Close();
        }

        public delegate void UpdateListCallBack(DataGridView list, List<Flight> data);
        private void UpdateListBox(DataGridView list, List<Flight> data)
        {
            list.Rows.Clear();
            List<DTOAirportFlight> dtoAirportFlights = updateLists(data);

            foreach (DTOAirportFlight dtoAirportFlight in dtoAirportFlights)
            {
                string[] row = { dtoAirportFlight.ID.ToString(), dtoAirportFlight.DepartureCity, dtoAirportFlight.DepartureName,
                    dtoAirportFlight.DestinationCity, dtoAirportFlight.DestinationName, dtoAirportFlight.DepartureDate.ToString("dd/MM/yyyy"),
                    dtoAirportFlight.DepartureTime.ToString("HH:mm"), dtoAirportFlight.FreeSeats.ToString() };

                list.Rows.Add(row);
            }
        }


        private void EmployeeUpdate(object sender, EmployeeEventArgs e)
        {
            if (e.Type != EmployeeEvent.BUY_TICKET)
            {
                return;
            }

            List<Flight> flights = e.Data as List<Flight>;
            allFlightsTable.BeginInvoke(new UpdateListCallBack(UpdateListBox), new Object[] { allFlightsTable, flights });
        }

        private void logOutButton_Click(object sender, EventArgs e)
        {
            controller.logout();
            controller.UpdateEvent -= EmployeeUpdate;
            Hide();
            logInForm.Show();
        }

        private void buyTicketsButton_Click(object sender, EventArgs e)
        {
            int id = int.Parse(searchFlightsTable.CurrentRow.Cells["searchIdColumn"].FormattedValue.ToString());
            updateSearchTable(controller.Service.findFlightByDestinationDate("", DateTime.Now));
            buyTicketForm = new BuyTicketForm(controller, controller.Service.findFlightById(id), this);
            buyTicketForm.Show();
            destinationTextBox.Text = "";
        }

        private void destinationTextBox_TextChanged(object sender, EventArgs e)
        {
            buyTicketsButton.Enabled = false;
            //updateSearchTable();
        }

        private void datePicker_ValueChanged(object sender, EventArgs e)
        {
            buyTicketsButton.Enabled = false;
            //updateSearchTable();
        }

        private void searchFlightsTable_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {
            buyTicketsButton.Enabled = true;
        }

        private void searchButton_Click(object sender, EventArgs e)
        {
            string destination = destinationTextBox.Text;
            DateTime dateTime = datePicker.Value;

            updateSearchTable(controller.Service.findFlightByDestinationDate(destination, dateTime));
        }
    }
}
