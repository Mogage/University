using Proiect_MPP.domain.people;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.repository.people.employees
{
    public interface IEmployeeRepository : IPersonRepository<Employee>
    {
        Employee? findByEmail(string email);
        List<Employee> getByPosition(string position);
    }
}
