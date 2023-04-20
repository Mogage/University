package root.persistence.invoices;

import root.model.Invoice;
import root.persistence.AbstractRepository;

import java.sql.*;
import java.util.Collection;
import java.util.Properties;

public class InvoiceRepository extends AbstractRepository<Invoice, Integer> implements IInvoiceRepository {
    public InvoiceRepository(Properties properties) {
        super(properties);
        logger.info("Initializing Invoice Repository ");
    }

    @Override
    protected Invoice extractEntity(ResultSet resultSet) throws SQLException {
        int id = resultSet.getInt("id");
        int clientId = resultSet.getInt("client_id");
        return new Invoice(id, clientId);
    }

    @Override
    protected PreparedStatement createInsertStatement(Connection connection, Invoice entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand, Statement.RETURN_GENERATED_KEYS);
        //statement.setInt(1, entity.getId());
        statement.setInt(1, entity.getClientId());
        return statement;
    }

    @Override
    protected PreparedStatement createUpdateStatement(Connection connection, Invoice entity, Integer integer) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        statement.setInt(2, entity.getId());
        statement.setInt(1, entity.getClientId());
        return statement;
    }

    @Override
    protected PreparedStatement createDeleteStatement(Connection connection, Invoice entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        statement.setInt(1, entity.getId());
        return statement;
    }

    @Override
    public int add(Invoice elem) {
        super.sqlCommand = "INSERT INTO invoices(client_id) VALUES (?)";
        return super.add(elem);
    }

    @Override
    public void delete(Invoice elem) {
        super.sqlCommand = "DELETE FROM invoices WHERE id=?";
        super.delete(elem);
    }

    @Override
    public void update(Invoice elem, Integer integer) {
        super.sqlCommand = "UPDATE invoices SET client_id=? WHERE id=?";
        super.update(elem, integer);
    }

    @Override
    public Invoice findById(Integer id) {
        super.sqlCommand = "SELECT * FROM invoices WHERE id="+id;
        return super.getOne();
    }

    @Override
    public Collection<Invoice> getAll() {
        super.sqlCommand = "SELECT * FROM invoices";
        return super.getAll();
    }
}
