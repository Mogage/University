using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Facultativ.Domain
{
    internal class Team : Entity<int>
    {
        private string name;
        public string Name 
        { 
            get { return name; } 
            set { name = value; } 
        }

        public Team(int id, string name) : base(id)
        {
            this.name = name;
        }

        public override string ToString()
        {
            return name;
        }
    }
}
