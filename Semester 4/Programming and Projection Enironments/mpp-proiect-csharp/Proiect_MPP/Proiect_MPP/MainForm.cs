using Proiect_MPP.domain;
using Proiect_MPP.repository.airports;
using Proiect_MPP.repository.flights;
using Proiect_MPP.service.main;
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
    public partial class MainForm : Form
    {
        IMainService mainService;
        IDictionary<string, string> properties;
        BuyTicketForm buyTicketForm;
        LogInForm logInForm;

        public MainForm(IDictionary<string, string> properties, LogInForm logInForm)
        {
            InitializeComponent();
            this.properties = properties;
            this.logInForm = logInForm;
            mainService = new MainService(new FlightRepository(properties), new AirportRepository(properties));
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            updateFlightsTable();
            updateSearchTable();
            buyTicketsButton.Enabled = false;
        }

        private IEnumerable<DTOAirportFlight> updateLists(IEnumerable<Flight> flights)
        {
            List<DTOAirportFlight> dTOAirportFlights = new List<DTOAirportFlight>();
            Airport departure;
            Airport destination;

            foreach (Flight flight in flights)
            {
                departure = mainService.findAirportById(flight.DepartureAirport);
                destination = mainService.findAirportById(flight.DestinationAirport);
                dTOAirportFlights.Add(new DTOAirportFlight(flight.ID, departure.CityName, departure.Name, destination.CityName,
                    destination.Name, flight.DepartureDate, flight.DepartureTime, flight.FreeSeats));
            }


            return dTOAirportFlights;
        }

        private void updateTable(DataGridView table, IEnumerable<DTOAirportFlight> dtoAirportFlights)
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

        private void updateFlightsTable()
        {
            IEnumerable<Flight> flights = mainService.getAllAvailableFlights();
            IEnumerable<DTOAirportFlight> dtoAirportFlights = updateLists(flights);
            updateTable(allFlightsTable, dtoAirportFlights);
        }

        private void updateSearchTable()
        {
            string destination = destinationTextBox.Text;
            DateTime dateTime = datePicker.Value;
            DateOnly date = new DateOnly(dateTime.Year, dateTime.Month, dateTime.Day);
            IEnumerable<Flight> flights = mainService.findByDestinationDate(destination, date);
            IEnumerable<DTOAirportFlight> dtoAirportFlights = updateLists(flights);
            updateTable(searchFlightsTable, dtoAirportFlights);
        }

        public void updateFlights(int flightId, int numberOfSeats)
        {
            Flight flight = mainService.findFlightById(flightId);
            flight.FreeSeats = numberOfSeats;
            mainService.updateFlight(flight, flightId);
            updateFlightsTable();
            updateSearchTable();
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);

            logInForm.Close();
        }

        private void logOutButton_Click(object sender, EventArgs e)
        {
            Hide();
            logInForm.Show();
        }

        private void buyTicketsButton_Click(object sender, EventArgs e)
        {
            int id = int.Parse(searchFlightsTable.CurrentRow.Cells["searchIdColumn"].FormattedValue.ToString());
            buyTicketForm = new BuyTicketForm(properties, mainService.findFlightById(id), this);
            buyTicketForm.Show();
        }

        private void destinationTextBox_TextChanged(object sender, EventArgs e)
        {
            buyTicketsButton.Enabled = false;
            updateSearchTable();
        }

        private void datePicker_ValueChanged(object sender, EventArgs e)
        {
            buyTicketsButton.Enabled = false;
            updateSearchTable();
        }

        private void searchFlightsTable_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {
            buyTicketsButton.Enabled = true;
        }
    }
}
