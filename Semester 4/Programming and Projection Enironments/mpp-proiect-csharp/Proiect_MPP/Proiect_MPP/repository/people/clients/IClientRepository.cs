using Proiect_MPP.domain.people;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.people.clients
{
    public interface IClientRepository : IPersonRepository<Client>
    {
        Client? findByAddress(string address);
    }
}
