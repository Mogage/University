using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Facultativ.Domain
{
    internal class Player : Student
    {
        private Team team;
        public Team Team
        {
            get { return team; }
            set { team = value; }
        }

        public Player(int id, string name, string school, Team team) : base(id, name, school)
        {
            this.team = team;
        }

        public override string ToString()
        {
            return base.ToString() + " - " + team.Name;
        }
    }
}
