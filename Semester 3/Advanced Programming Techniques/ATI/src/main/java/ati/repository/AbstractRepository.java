package ati.repository;

import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public abstract class AbstractRepository<T> {
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
}
