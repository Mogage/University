﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace model
{
    [Serializable]
    public class DTOAirportFlight : Entity<int>
    {
        private string departureName;
        private string departureCity;
        private string destinationName;
        private string destinationCity;
        private DateTime departureDate;
        private DateTime departureTime;
        private int freeSeats;

        public DTOAirportFlight(int id, string departureName, string departureCity, string destinationName, string destinationCity,
            DateTime departureDate, DateTime departureTime, int freeSeats) : base(id)
        {
            this.departureName = departureName;
            this.departureCity = departureCity;
            this.destinationName = destinationName;
            this.destinationCity = destinationCity;
            this.departureDate = departureDate;
            this.departureTime = departureTime;
            this.freeSeats = freeSeats;
        }

        public string DepartureName { get => departureName; set => departureName = value; }
        public string DepartureCity { get => departureCity; set => departureCity = value; }
        public string DestinationName { get => destinationName; set => destinationName = value; }
        public string DestinationCity { get => destinationCity; set => destinationCity = value; }
        public DateTime DepartureDate { get => departureDate; set => departureDate = value; }
        public DateTime DepartureTime { get => departureTime; set => departureTime = value; }
        public int FreeSeats { get => freeSeats; set => freeSeats = value; }
    }
}
