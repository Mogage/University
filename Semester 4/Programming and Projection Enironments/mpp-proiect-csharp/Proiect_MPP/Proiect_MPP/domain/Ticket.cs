using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.domain
{
    internal class Ticket : Entity<int>
    {
        private int flightId;
        private int invoiceId;
        private int seatNumber;
        private string touristName;

        // Class Constructors //

        public Ticket() : base(0)
        {
            this.flightId = 0;
            this.invoiceId = 0;
            this.seatNumber = 0;
            this.touristName = "";
        }

        public Ticket(int flightId, int invoiceId, int seatNumber, String touristName) : base(0)
        {
            this.flightId = flightId;
            this.invoiceId = invoiceId;
            this.seatNumber = seatNumber;
            this.touristName = touristName;
        }

        public Ticket(int id, int flightId, int invoiceId, int seatNumber, String touristName) : base(id)
        {
            this.flightId = flightId;
            this.invoiceId = invoiceId;
            this.seatNumber = seatNumber;
            this.touristName = touristName;
        }

        // Getters & Setters //

        public int FlightId
        {
            get { return this.flightId; }
            set { this.flightId = value; }
        }

        public int InvoiceId
        {
            get { return this.invoiceId; }
            set { this.invoiceId = value; }
        }

        public int SeatNumber
        {
            get { return this.seatNumber; }
            set { this.seatNumber = value; }
        }

        public string TouristName
        {
            get { return this.touristName; }
            set { this.touristName = value; }
        }

        // ToString & other functions //

        public override string ToString()
        {
            return "Ticket{" +
                 "id=" + base.ID +
                 ", flightId=" + flightId +
                 ", invoiceId=" + invoiceId +
                 ", seatNumber=" + seatNumber +
                 ", touristId=" + touristName +
                 '}';
        }
    }
}
