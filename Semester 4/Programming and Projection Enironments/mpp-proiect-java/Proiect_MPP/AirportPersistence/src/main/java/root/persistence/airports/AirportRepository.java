package root.persistence.airports;

import root.model.Airport;
import root.persistence.AbstractRepository;

import java.sql.*;
import java.util.Collection;
import java.util.Properties;

public class AirportRepository extends AbstractRepository<Airport, Integer> implements IAirportRepository {

    public AirportRepository(Properties properties) {
        super(properties);
        logger.info("Initializing Airport Repository ");
    }

    @Override
    protected Airport extractEntity(ResultSet resultSet) throws SQLException {
        int id = resultSet.getInt("id");
        String name = resultSet.getString("name");
        String cityName = resultSet.getString("city_name");
        return new Airport(id, name, cityName);
    }

    @Override
    protected PreparedStatement createInsertStatement(Connection connection, Airport entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand, Statement.RETURN_GENERATED_KEYS);
        statement.setInt(1, entity.getId());
        statement.setString(2, entity.getName());
        statement.setString(3, entity.getCityName());
        return statement;
    }

    @Override
    protected PreparedStatement createUpdateStatement(Connection connection, Airport entity, Integer integer) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        statement.setInt(3, entity.getId());
        statement.setString(1, entity.getName());
        statement.setString(2, entity.getCityName());
        return statement;
    }

    @Override
    protected PreparedStatement createDeleteStatement(Connection connection, Airport entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        statement.setLong(1, entity.getId());
        return statement;
    }

    @Override
    public int add(Airport elem) {
        super.sqlCommand = "INSERT INTO airports(id, name, city_name) VALUES (?, ?, ?)";
        return super.add(elem);
    }

    @Override
    public void delete(Airport elem) {
        super.sqlCommand = "DELETE FROM airports WHERE id=?";
        super.delete(elem);
    }

    @Override
    public void update(Airport elem, Integer integer) {
        super.sqlCommand = "UPDATE airports SET name=?, city_name=? WHERE id=?";
        super.update(elem, integer);
    }

    @Override
    public Airport findById(Integer integer) {
        super.sqlCommand = "SELECT * FROM airports WHERE id="+integer;
        return super.getOne();
    }

    @Override
    public Collection<Airport> getAll() {
        super.sqlCommand = "SELECT * FROM airports";
        return super.getAll();
    }

    @Override
    public Collection<Airport> getAirportAfterName(String name) {
        super.sqlCommand = "SELECT * FROM airports WHERE name='" + name + '\'';
        return super.getAll();
    }

    @Override
    public Collection<Airport> getAirportsInCity(String cityName) {
        super.sqlCommand = "SELECT * FROM airports WHERE LOWER(city_name) LIKE '" + cityName.toLowerCase() + "%'";
        logger.info(super.sqlCommand);
        return super.getAll();
    }

}
