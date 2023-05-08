using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace model.people
{
    [Serializable]
    public class Person : Entity<int>
    {
        private string firstName;
        private string lastName;

        // Class Constructors //

        public Person() : base(0)
        {
            this.firstName = "";
            this.lastName = "";
        }

        public Person(string firstName, string lastName) : base(0)
        {
            this.firstName = firstName;
            this.lastName = lastName;
        }

        public Person(int id, string firstName, string lastName) : base(id)
        {
            this.firstName = firstName;
            this.lastName = lastName;
        }

        // Getters & Setters //

        public string FirstName
        {
            get { return this.firstName; }
            set { this.firstName = value; }
        }

        public string LastName
        {
            get { return this.lastName; }
            set { this.lastName = value; }
        }

        // toString & other functions //

        public override string ToString() 
        {
            return "id=" + base.ID +
                ", firstName='" + firstName + '\'' +
                ", lastName='" + lastName + '\'';
        }
    }
}
