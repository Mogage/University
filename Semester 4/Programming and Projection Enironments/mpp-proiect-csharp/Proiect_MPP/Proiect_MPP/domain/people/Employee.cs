using Microsoft.VisualBasic.ApplicationServices;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.domain.people
{
    internal class Employee : Person
    {
        private string position;
        private string email;
        private string password;

        // Class Constructors //

        public Employee() : base()
        {
            this.position = "";
            this.email = "";
            this.password = "";
        }

        public Employee(string firstName, string lastName, string position, string email, string password) : base(firstName, lastName)
        {
            this.position = position;
            this.email = email;
            this.password = password;
        }

        public Employee(int id, string firstName, string lastName, string position, string email, string password) : base(id, firstName, lastName)
        {
            this.position = position;
            this.email = email;
            this.password = password;
        }

        // Getters & Setters //

        public string Position
        {
            get { return this.position; }
            set { this.position = value; }
        }

        public string Email
        {
            get { return this.email; }
            set { this.email = value; }
        }

        public string Password
        {
            get { return this.password; }
            set { this.password = value; }
        }

        // toString & other functions //

        public override string ToString()
        {
            return "Employee{" + base.ToString() +
                "position='" + position + '\'' +
                ", email='" + email + '\'' +
                ", password='" + password + '\'' +
                '}';
        }
    }
}
