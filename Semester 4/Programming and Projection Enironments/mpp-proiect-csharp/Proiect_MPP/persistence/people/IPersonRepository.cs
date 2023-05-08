using model.people;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistence.people
{
    public interface IPersonRepository<T> : Repository<T, int> where T : Person
    {
        List<T> getPersonByFirstName(string firstName);
        List<T> getPersonByLastName(string lastName);
    }
}
