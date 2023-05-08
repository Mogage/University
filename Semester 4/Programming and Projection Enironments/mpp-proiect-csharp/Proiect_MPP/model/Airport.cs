using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace model
{
    [Serializable]
    public class Airport : Entity<int>
    {
        private string name;
        private string cityName;

        // Class Constructors //

        public Airport() : base(0)
        {
            this.name = "";
            this.cityName = "";
        }

        public Airport(string name, string cityName) : base(0)
        {
            this.name = name;
            this.cityName = cityName;
        }

        public Airport(int id, string name, string cityName) : base(id)
        {
            this.name = name;
            this.cityName = cityName;
        }

        // Getters & Setters //

        public string Name
        {
            get { return name; }
            set { name = value; }
        }

        public string CityName
        {
            get { return cityName; }
            set { cityName = value; }
        }

        // ToString & other functions

        public override string ToString() 
        {
            return "Airport{" +
                "id=" + base.ID +
                ", name='" + name + '\'' +
                '}';
        }
    }
}
