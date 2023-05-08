package root.proiect_mpp.service.logIn;

import root.proiect_mpp.domain.people.Employee;
import root.proiect_mpp.repositories.people.employees.EmployeeRepository;

public class LogInService implements ILogInService {
    private final EmployeeRepository employeeRepository;

    public LogInService(EmployeeRepository employeeRepository) {
        this.employeeRepository = employeeRepository;
    }

    public Employee findByEmail(String email) {
        // Validate email
        return employeeRepository.findByEmail(email);
    }
}
