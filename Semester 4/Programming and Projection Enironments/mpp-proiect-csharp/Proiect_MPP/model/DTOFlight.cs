using model.people;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace model
{
    [Serializable]
    public class DTOFlight
    {
        public Client client { get; }
        public List<Person> people { get; }
        public Flight flight { get; }

        public DTOFlight(Client client, List<Person> people, Flight flight)
        {
            this.client = client;
            this.people = people;
            this.flight = flight;
        }

        public override string ToString()
        {
            return "DTOFlight{" +
                    "client=" + client +
                    ", people=" + people +
                    ", flight=" + flight +
                    '}';
        }
    }
}
