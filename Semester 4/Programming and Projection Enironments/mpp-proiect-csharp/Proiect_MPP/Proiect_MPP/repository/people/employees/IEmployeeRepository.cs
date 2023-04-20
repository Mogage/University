using Proiect_MPP.domain.people;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.people.employees
{
    internal interface IEmployeeRepository : IPersonRepository<Employee>
    {
        Employee? findByEmail(string email);
        IEnumerable<Employee> getByPosition(string position);
    }
}
