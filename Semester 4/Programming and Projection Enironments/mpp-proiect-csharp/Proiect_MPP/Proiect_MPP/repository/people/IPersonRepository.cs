using Proiect_MPP.domain.people;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.people
{
    internal interface IPersonRepository<T> : Repository<T, int> where T : Person
    {
        IEnumerable<T> getPersonByFirstName(string firstName);
        IEnumerable<T> getPersonByLastName(string lastName);
    }
}
