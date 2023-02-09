package examen.repository;

import examen.domain.Entity;

import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public abstract class AbstractRepository<ID, T extends Entity<ID>> {
    protected final String url;
    protected final String userName;
    protected final String password;
    protected String sqlCommand;

    public AbstractRepository(String url, String userName, String password) {
        this.url = url;
        this.userName = userName;
        this.password = password;
        this.sqlCommand = "";
    }

    protected abstract T extractEntity(ResultSet resultSet) throws SQLException;

    protected abstract PreparedStatement createStatementFromEntity(Connection connection, T entity) throws SQLException;

    public Iterable<T> getAll() {
        List<T> entities = new ArrayList<>();
        try (Connection connection = DriverManager.getConnection(url, userName, password);
             PreparedStatement statement = connection.prepareStatement(sqlCommand);
             ResultSet resultSet = statement.executeQuery()
        ) {
            while (resultSet.next()) {
                T entity = extractEntity(resultSet);
                entities.add(entity);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return entities;
    }

    public void save(T entity) throws Exception {
        try (Connection connection = DriverManager.getConnection(url, userName, password);
             PreparedStatement statement = createStatementFromEntity(connection, entity)
        ) {
            statement.executeUpdate();
        } catch (SQLException e) {
            throw new Exception(e.getMessage());
        }
    }

    public T findAfterId(ID id) {
        try (Connection connection = DriverManager.getConnection(url, userName, password);
             PreparedStatement statement = connection.prepareStatement(sqlCommand);
             ResultSet resultSet = statement.executeQuery();
        ) {
            if(resultSet.next())
                return extractEntity(resultSet);
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return null;
    }
}