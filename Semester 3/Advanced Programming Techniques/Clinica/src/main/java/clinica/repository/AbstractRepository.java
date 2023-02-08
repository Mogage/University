package clinica.repository;

import clinica.domain.Entity;

import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public abstract class AbstractRepository<ID, T extends Entity<ID>> implements Repository<ID, T> {
    private final String url;
    private final String userName;
    private final String password;
    private String sqlCommand;

    public AbstractRepository(String url, String userName, String password, String sqlCommand) {
        this.url = url;
        this.userName = userName;
        this.password = password;
        this.sqlCommand = sqlCommand;
    }

    public void setSqlCommand(String sqlCommand) {
        this.sqlCommand = sqlCommand;
    }

    protected abstract T extractEntity(ResultSet resultSet) throws SQLException;

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
