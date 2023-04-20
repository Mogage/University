package root.proiect_mpp.service.logIn;

import root.proiect_mpp.domain.people.Employee;

public interface ILogInService {
    Employee findByEmail(String email);
}
