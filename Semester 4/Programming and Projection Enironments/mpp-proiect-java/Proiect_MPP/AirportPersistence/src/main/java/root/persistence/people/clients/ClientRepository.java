package root.persistence.people.clients;

import root.model.people.Client;
import root.persistence.people.PersonRepository;

import java.sql.*;
import java.util.Collection;
import java.util.Properties;


public class ClientRepository extends PersonRepository<Client> implements IClientRepository {

    public ClientRepository(Properties properties) {
        super(properties);
        logger.info("Initializing Client Repository ");
    }

    @Override
    protected String getTableName() {
        return "clients";
    }

    @Override
    protected Client extractEntity(ResultSet resultSet) throws SQLException {
        int id = resultSet.getInt("id");
        String firstName = resultSet.getString("first_name");
        String lastName = resultSet.getString("last_name");
        String address = resultSet.getString("address");
        return new Client(id, firstName, lastName, address);
    }

    @Override
    protected PreparedStatement createInsertStatement(Connection connection, Client entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand, Statement.RETURN_GENERATED_KEYS);
        //statement.setInt(1, entity.getId());
        statement.setString(1, entity.getFirstName());
        statement.setString(2, entity.getLastName());
        statement.setString(3, entity.getAddress());
        return statement;
    }

    @Override
    protected PreparedStatement createUpdateStatement(Connection connection, Client entity, Integer integer) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        statement.setInt(4, entity.getId());
        statement.setString(1, entity.getFirstName());
        statement.setString(2, entity.getLastName());
        statement.setString(3, entity.getAddress());
        return statement;
    }

    @Override
    protected PreparedStatement createDeleteStatement(Connection connection, Client entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        statement.setInt(1, entity.getId());
        return statement;
    }

    @Override
    public int add(Client elem) {
        super.sqlCommand = "INSERT INTO clients(first_name, last_name, address) VALUES (?, ?, ?)";
        return super.add(elem);
    }

    @Override
    public void delete(Client elem) {
        super.sqlCommand = "DELETE FROM clients WHERE id=?";
        super.delete(elem);
    }

    @Override
    public void update(Client elem, Integer integer) {
        super.sqlCommand = "UPDATE clients SET first_name=?, last_name=?, address=? WHERE id=?";
        super.update(elem, integer);
    }

    @Override
    public Client findById(Integer integer) {
        super.sqlCommand = "SELECT * FROM clients WHERE id=" + integer;
        return super.getOne();
    }

    @Override
    public Collection<Client> getAll() {
        super.sqlCommand = "SELECT * FROM clients";
        return super.getAll();
    }

    @Override
    public Client findByAddress(String address) {
        super.sqlCommand = "SELECT * FROM clients WHERE address='" + address + '\'';
        return super.getOne();
    }
}
