using Proiect_MPP.domain.people;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.people
{
    public interface IPersonRepository<T> : Repository<T, int> where T : Person
    {
        List<T> getPersonByFirstName(string firstName);
        List<T> getPersonByLastName(string lastName);
    }
}
