package root.proiect_mpp.repositories.people.employees;

import root.proiect_mpp.domain.people.Employee;
import root.proiect_mpp.repositories.people.PersonRepository;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collection;
import java.util.Properties;

public class EmployeeRepository extends PersonRepository<Employee> implements IEmployeeRepository {
    public EmployeeRepository(Properties properties) {
        super(properties);
        logger.info("Initializing Employee Repository ");
    }

    @Override
    protected String getTableName() {
        return "employees";
    }

    @Override
    protected Employee extractEntity(ResultSet resultSet) throws SQLException {
        int id = resultSet.getInt("id");
        String firstName = resultSet.getString("first_name");
        String lastName = resultSet.getString("last_name");
        String position = resultSet.getString("position");
        String email = resultSet.getString("email");
        String password = resultSet.getString("password");
        return new Employee(id, firstName, lastName, position, email, password);
    }

    @Override
    protected PreparedStatement createInsertStatement(Connection connection, Employee entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        // statement.setInt(1, entity.getId());
        statement.setString(1, entity.getFirstName());
        statement.setString(2, entity.getLastName());
        statement.setString(3, entity.getPosition());
        statement.setString(4, entity.getEmail());
        statement.setString(5, entity.getPassword());
        return statement;
    }

    @Override
    protected PreparedStatement createUpdateStatement(Connection connection, Employee entity, Integer integer) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        statement.setInt(6, entity.getId());
        statement.setString(1, entity.getFirstName());
        statement.setString(2, entity.getLastName());
        statement.setString(3, entity.getPosition());
        statement.setString(4, entity.getEmail());
        statement.setString(5, entity.getPassword());
        return statement;
    }

    @Override
    protected PreparedStatement createDeleteStatement(Connection connection, Employee entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        statement.setInt(1, entity.getId());
        return statement;
    }

    @Override
    public void add(Employee elem) {
        super.sqlCommand = "INSERT INTO employees (first_name, last_name, position, email, password) VALUES (?, ?, ?, ?, ?)";
        super.add(elem);
    }

    @Override
    public void delete(Employee elem) {
        super.sqlCommand = "DELETE FROM employees WHERE id=?";
        super.delete(elem);
    }

    @Override
    public void update(Employee elem, Integer integer) {
        super.sqlCommand = "UPDATE employees SET first_name=?, last_name=?, position=?, email=?, password=? WHERE id=?";
        super.update(elem, integer);
    }

    @Override
    public Employee findById(Integer integer) {
        super.sqlCommand = "SELECT * FROM employees WHERE id=" + integer;
        return super.getOne();
    }

    @Override
    public Employee findByEmail(String email) {
        super.sqlCommand = "SELECT * FROM employees WHERE email='" + email + '\'';
        return super.getOne();
    }

    @Override
    public Collection<Employee> getAll() {
        super.sqlCommand = "SELECT * FROM employees";
        return super.getAll();
    }

    @Override
    public Collection<Employee> getByPosition(String position) {
        super.sqlCommand = "SELECT * FROM employees WHERE position='" + position + '\'';
        return super.getAll();
    }
}
