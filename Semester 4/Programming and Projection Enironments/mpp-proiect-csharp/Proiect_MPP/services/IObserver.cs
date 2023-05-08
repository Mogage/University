using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using model;

namespace services
{
    public interface IObserver
    {
        void ticketBought(List<Flight> flights);
    }
}
