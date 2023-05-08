using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace model
{
    [Serializable]
    public class DTODestinationDate
    {
        public string Destination { get; }
        public DateTime Date { get; }

        public DTODestinationDate(string destination, DateTime date)
        {
            Destination = destination;
            Date = date;
        }

        public override string ToString()
        {
            return "DTODestinationDate{" +
                "destination='" + Destination + '\'' +
                ", date=" + Date.ToString() +
                '}';
        }
    }
}
