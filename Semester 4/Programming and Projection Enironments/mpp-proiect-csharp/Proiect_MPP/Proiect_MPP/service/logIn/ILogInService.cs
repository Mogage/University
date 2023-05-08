using Proiect_MPP.domain.people;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proiect_MPP.service.logIn
{
    public interface ILogInService
    {
        Employee findByEmail(string email);
    }
}
