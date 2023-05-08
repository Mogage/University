using model.people;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistence.people.clients
{
    public interface IClientRepository : IPersonRepository<Client>
    {
        Client? findByAddress(string address);
    }
}
