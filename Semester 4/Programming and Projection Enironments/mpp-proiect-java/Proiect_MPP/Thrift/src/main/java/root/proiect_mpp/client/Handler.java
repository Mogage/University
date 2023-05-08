package root.proiect_mpp.client;

import root.proiect_mpp.repositories.people.employees.EmployeeRepository;
import root.proiect_mpp.domain.people.Employee;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

public class Handler implements Service.Iface{
    EmployeeRepository employeeRepository;
    private final Map<Integer, Integer> loggedEmployees;

    public Handler(Properties props){
        employeeRepository = new EmployeeRepository(props);
        loggedEmployees = new ConcurrentHashMap<>();
    }

    @Override
    public EmployeeThrift login(String username, String password) throws org.apache.thrift.TException {
        Employee employee = new Employee(username, password);
        Employee employeeToLogin = employeeRepository.findByEmail(employee.getEmail());
        EmployeeThrift employeeThrift = new EmployeeThrift();
        employeeThrift.id = employeeToLogin.getId();
        employeeThrift.firstName = employeeToLogin.getFirstName();
        employeeThrift.lastName = employeeToLogin.getLastName();
        employeeThrift.position = employeeToLogin.getPosition();
        employeeThrift.email = employeeToLogin.getEmail();
        employeeThrift.password = employeeToLogin.getPassword();
        if (employeeToLogin != null) {
            if (loggedEmployees.get(employeeToLogin.getId()) != null)
                throw new org.apache.thrift.TException("Employee already logged in.");

            //loggedEmployees.put(employeeToLogin.getId(), client);
            return employeeThrift;
        } else {
            throw new org.apache.thrift.TException("Authentication failed.");
        }
    }
}
