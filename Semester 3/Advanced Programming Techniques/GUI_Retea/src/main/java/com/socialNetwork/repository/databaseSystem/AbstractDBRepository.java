package com.socialNetwork.repository.databaseSystem;

import com.socialNetwork.domain.Entity;
import com.socialNetwork.exceptions.RepositoryException;
import com.socialNetwork.repository.InMemoryRepository;

import java.sql.*;

public abstract class AbstractDBRepository<ID, T extends Entity<ID>> extends InMemoryRepository<ID, T> {
    private final String url;
    private final String userName;
    private final String password;
    private String sqlCommand;

    public AbstractDBRepository(String url, String userName, String password, String sqlCommand) {
        super();
        this.url = url;
        this.userName = userName;
        this.password = password;
        this.sqlCommand = sqlCommand;
        loadData(this.sqlCommand);
    }

    public String getSqlCommand() {
        return sqlCommand;
    }

    public void setSqlCommand(String sqlCommand) {
        this.sqlCommand = sqlCommand;
    }

    public void clearData() {
        super.entities.clear();
    }

    public void loadData(String sqlCommand) {
        try (Connection connection = DriverManager.getConnection(url, userName, password);
             PreparedStatement statement = connection.prepareStatement(sqlCommand);
             ResultSet resultSet = statement.executeQuery()
        ) {
            while (resultSet.next()) {
                T entity = extractEntity(resultSet);
                super.save(entity);
            }
        } catch (SQLException | RepositoryException e) {
            e.printStackTrace();
        }
    }

    protected abstract T extractEntity(ResultSet resultSet) throws SQLException;

    protected abstract PreparedStatement createInsertStatement(Connection connection, T entity) throws SQLException;

    protected abstract PreparedStatement createUpdateStatement(Connection connection, T entity) throws SQLException;

    protected abstract PreparedStatement createDeleteStatement(Connection connection, T entity) throws SQLException;

    protected PreparedStatement createStatementFromEntity(Connection connection, T entity) throws SQLException {
        char command = sqlCommand.charAt(0);
        return switch (command) {
            case 'I' -> createInsertStatement(connection, entity);
            case 'U' -> createUpdateStatement(connection, entity);
            case 'D' -> createDeleteStatement(connection, entity);
            default -> null;
        };
    }

    private void executeCommand(T entity) throws RepositoryException {
        try (Connection connection = DriverManager.getConnection(url, userName, password);
             PreparedStatement statement = createStatementFromEntity(connection, entity)
        ) {
            statement.executeUpdate();
        } catch (SQLException e) {
            throw new RepositoryException(e.getMessage());
        }
    }

    @Override
    public void save(T entity) throws IllegalArgumentException, RepositoryException {
        super.save(entity);
        executeCommand(entity);
    }

    @Override
    public void update(ID id, T entity) throws IllegalArgumentException, RepositoryException {
        super.update(id, entity);
        executeCommand(entity);
    }

    @Override
    public T delete(ID id) throws RepositoryException {
        T deleted = super.delete(id);
        executeCommand(deleted);
        return deleted;
    }
}
