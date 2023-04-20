package root.proiect_mpp.repositories.flights;

import root.proiect_mpp.domain.Flight;
import root.proiect_mpp.repositories.AbstractRepository;
import root.proiect_mpp.utils.Constants;

import java.lang.invoke.ConstantBootstraps;
import java.sql.*;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.Collection;
import java.util.Properties;


public class FlightRepository extends AbstractRepository<Flight, Integer> implements IFlightRepository {
    public FlightRepository(Properties properties) {
        super(properties);
        logger.info("Initializing Flight Repository ");
    }

    @Override
    protected Flight extractEntity(ResultSet resultSet) throws SQLException {
        int id = resultSet.getInt("id");
        int freeSeats = resultSet.getInt("free_seats");
        int destinationAirport = resultSet.getInt("destination_airport");
        int departureAirport = resultSet.getInt("departure_airport");
        LocalDate localDate = LocalDate.parse(resultSet.getString("departure_date"), Constants.DATE_FORMATTER);
        LocalTime localTime = LocalTime.parse(resultSet.getString("departure_time"), Constants.TIME_FORMATTER);
        return new Flight(id, freeSeats, destinationAirport, departureAirport, localDate, localTime);
    }

    @Override
    protected PreparedStatement createInsertStatement(Connection connection, Flight entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand, Statement.RETURN_GENERATED_KEYS);
        statement.setInt(1, entity.getId());
        statement.setInt(2, entity.getFreeSeats());
        statement.setInt(3, entity.getDestinationAirport());
        statement.setInt(4, entity.getDepartureAirport());
        statement.setString(5, entity.getDepartureDate().format(Constants.DATE_FORMATTER));
        statement.setString(6, entity.getDepartureTime().format(Constants.TIME_FORMATTER));
        return statement;
    }

    @Override
    protected PreparedStatement createUpdateStatement(Connection connection, Flight entity, Integer integer) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        statement.setInt(6, entity.getId());
        statement.setInt(1, entity.getFreeSeats());
        statement.setInt(2, entity.getDestinationAirport());
        statement.setInt(3, entity.getDepartureAirport());
        statement.setString(4, entity.getDepartureDate().format(Constants.DATE_FORMATTER));
        statement.setString(5, entity.getDepartureTime().format(Constants.TIME_FORMATTER));
        return statement;
    }

    @Override
    protected PreparedStatement createDeleteStatement(Connection connection, Flight entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        statement.setInt(1, entity.getId());
        return statement;
    }

    @Override
    public int add(Flight elem) {
        super.sqlCommand = "INSERT INTO flights(id, free_seats, destination_airport, departure_airport, departure_date, " +
                "departure_time) VALUES (?, ?, ?, ?, ?, ?)";
        return super.add(elem);
    }

    @Override
    public void delete(Flight elem) {
        super.sqlCommand = "DELETE FROM flights WHERE id=?";
        super.delete(elem);
    }

    @Override
    public void update(Flight elem, Integer integer) {
        super.sqlCommand = "UPDATE flights SET free_seats=?, destination_airport=?, departure_airport=?, departure_date=?, " +
                "departure_time=? WHERE id=?";
        super.update(elem, integer);
    }

    @Override
    public Flight findById(Integer id) {
        super.sqlCommand = "SELECT * FROM flights WHERE id=" + id;
        return super.getOne();
    }

    @Override
    public Collection<Flight> getAll() {
        super.sqlCommand = "SELECT * FROM flights";
        return super.getAll();
    }

    @Override
    public Collection<Flight> getAvailable() {
        super.sqlCommand = "SELECT * FROM flights WHERE free_seats <> 0";
        return super.getAll();
    }

    @Override
    public Collection<Flight> getByDepartureAirport(int departureAirport) {
        super.sqlCommand = "SELECT * FROM flights WHERE departure_airport=" + departureAirport;
        return super.getAll();
    }

    @Override
    public Collection<Flight> getByDestinationAirport(int destinationAirport) {
        super.sqlCommand = "SELECT * FROM flights WHERE destination_airport=" + destinationAirport;
        logger.info(super.sqlCommand);
        return super.getAll();
    }

    @Override
    public Collection<Flight> getAfterDepartureDateTime(LocalDateTime departureDateTime) {
        super.sqlCommand = "SELECT * FROM flights WHERE departure_date>='" + departureDateTime.format(Constants.DATE_FORMATTER) +
                "' AND departure_time>='" + departureDateTime.format(Constants.TIME_FORMATTER) + '\'';
        return super.getAll();
    }
}
