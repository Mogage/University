using model.people;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistence.people.employees
{
    public interface IEmployeeRepository : IPersonRepository<Employee>
    {
        Employee? findByEmail(string email);
        List<Employee> getByPosition(string position);
    }
}
