using Proiect_MPP.domain.people;
using Proiect_MPP.repository.people.employees;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.service.logIn
{
    public class LogInService : ILogInService
    {
        private readonly IEmployeeRepository _employeeRepository;
        public LogInService(IEmployeeRepository employeeRepository)
        {
            _employeeRepository = employeeRepository;
        }
        public Employee findByEmail(string email)
        {
            return _employeeRepository.findByEmail(email);
        }
    }
}
