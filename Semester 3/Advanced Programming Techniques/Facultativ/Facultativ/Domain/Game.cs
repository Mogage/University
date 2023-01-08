using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Facultativ.Domain
{
    internal class Game : Entity<int>
    {
        private Team hostTeam;
        private Team guestTeam;
        private DateTime date;

        public Team HostTeam
        {
            get { return hostTeam; }
            set { hostTeam = value; }
        }
        public Team GuestTeam
        {
            get { return guestTeam; }
            set { guestTeam = value; }
        }
        public DateTime Date
        {
            get { return date; }
            set { date = value; }
        }

        public Game(int id, Team hostTeam, Team guestTeam, DateTime date) : base(id)
        {
            this.hostTeam = hostTeam;
            this.guestTeam = guestTeam;
            this.date = date;
        }

        public override string ToString() 
        { 
            return hostTeam.ToString() + " - " + guestTeam.ToString() + "\t at " + date.ToString();
        }
    }
}
