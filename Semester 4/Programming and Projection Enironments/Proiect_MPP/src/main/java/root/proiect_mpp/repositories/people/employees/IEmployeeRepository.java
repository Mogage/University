package root.proiect_mpp.repositories.people.employees;

import root.proiect_mpp.domain.people.Employee;
import root.proiect_mpp.repositories.people.IPersonRepository;

import java.util.Collection;

public interface IEmployeeRepository extends IPersonRepository<Employee> {
    Employee findByEmail(String email);
    Collection<Employee> getByPosition(String position);
}
