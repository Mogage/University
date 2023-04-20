package root.persistence.people.employees;

import root.model.people.Employee;
import root.persistence.people.IPersonRepository;

import java.util.Collection;

public interface IEmployeeRepository extends IPersonRepository<Employee> {
    Employee findByEmail(String email);
    Collection<Employee> getByPosition(String position);
}
