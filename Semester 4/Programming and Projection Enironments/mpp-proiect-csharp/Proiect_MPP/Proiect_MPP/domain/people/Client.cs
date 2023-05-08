using Microsoft.VisualBasic.ApplicationServices;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.domain.people
{
    public class Client : Person
    {
        private string address;

        // Class constructors //

        public Client() : base()
        {
            address = "";
        }

        public Client(string firstName, string lastName, string address) : base(firstName, lastName)
        {
            this.address = address;
        }

        public Client(int id, string firstName, string lastName, string address) : base(id, firstName, lastName)
        {
            this.address = address;
        }

        // Getters & Setters //

        public string Address
        {
            get { return this.address; }
            set { this.address = value; }
        }

        // toString & other functions

        public override string ToString() 
        {
            return "Client{" + base.ToString() +
                    "address='" + address + '\'' +
                    '}';
        }
    }
}
