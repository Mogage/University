package root.persistence.tickets;

import root.model.Ticket;
import root.persistence.AbstractRepository;

import java.sql.*;
import java.util.Collection;
import java.util.Properties;

public class TicketRepository extends AbstractRepository<Ticket, Integer> implements ITicketRepository {
    public TicketRepository(Properties properties) {
        super(properties);
        logger.info("Initializing Ticket Repository ");
    }

    @Override
    protected Ticket extractEntity(ResultSet resultSet) throws SQLException {
        int id = resultSet.getInt("id");
        int flightId = resultSet.getInt("flight_id");
        int invoiceId = resultSet.getInt("invoice_id");
        int seatNumber = resultSet.getInt("seat_number");
        String touristName = resultSet.getString("tourist_name");
        return new Ticket(id, flightId, invoiceId, seatNumber, touristName);
    }

    @Override
    protected PreparedStatement createInsertStatement(Connection connection, Ticket entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand, Statement.RETURN_GENERATED_KEYS);
        //statement.setInt(1, entity.getId());
        statement.setInt(1, entity.getFlightId());
        statement.setInt(2, entity.getInvoiceId());
        statement.setInt(3, entity.getSeatNumber());
        statement.setString(4, entity.getTouristName());
        return statement;
    }

    @Override
    protected PreparedStatement createUpdateStatement(Connection connection, Ticket entity, Integer integer) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        statement.setInt(5, entity.getId());
        statement.setInt(1, entity.getFlightId());
        statement.setInt(2, entity.getInvoiceId());
        statement.setInt(3, entity.getSeatNumber());
        statement.setString(4, entity.getTouristName());
        return statement;
    }

    @Override
    protected PreparedStatement createDeleteStatement(Connection connection, Ticket entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        statement.setInt(1, entity.getId());
        return statement;
    }

    @Override
    public int add(Ticket elem) {
        super.sqlCommand = "INSERT INTO tickets (flight_id, invoice_id, seat_number, tourist_name) VALUES (?, ?, ?, ?)";
        return super.add(elem);
    }

    @Override
    public void delete(Ticket elem) {
        super.sqlCommand = "DELETE FROM tickets WHERE id=?";
        super.delete(elem);
    }

    @Override
    public void update(Ticket elem, Integer integer) {
        super.sqlCommand = "UPDATE tickets SET flight_id=?, invoice_id=?, seat_number=?, tourist_name=? WHERE id=?";
        super.update(elem, integer);
    }

    @Override
    public Ticket findById(Integer id) {
        super.sqlCommand = "SELECT * FROM tickets WHERE id=" + id;
        return super.getOne();
    }

    @Override
    public Ticket findByTouristName(String touristName) {
        super.sqlCommand = "SELECT * FROM tickets WHERE tourist_name='" + touristName + '\'';
        return super.getOne();
    }

    @Override
    public Collection<Ticket> getAll() {
        super.sqlCommand = "SELECT * FROM tickets";
        return super.getAll();
    }

    @Override
    public Collection<Ticket> getByInvoiceId(int invoiceID) {
        super.sqlCommand = "SELECT * FROM tickets WHERE invoice_id="+invoiceID;
        return super.getAll();
    }

    @Override
    public Collection<Ticket> getByFlightId(int flightID) {
        super.sqlCommand = "SELECT * FROM tickets WHERE flight_id=" + flightID;
        return super.getAll();
    }
}

