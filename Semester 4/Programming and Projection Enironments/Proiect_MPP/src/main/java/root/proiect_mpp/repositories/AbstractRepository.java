package root.proiect_mpp.repositories;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import root.proiect_mpp.domain.Entity;
import root.proiect_mpp.utils.JdbcUtils;

import java.sql.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Properties;

public abstract class AbstractRepository<T extends Entity<ID>, ID> implements Repository<T, ID> {
    private final JdbcUtils dbUtils;
    protected String sqlCommand;
    protected static final Logger logger = LogManager.getLogger();

    // Class constructors //

    public AbstractRepository(Properties properties) {
        logger.info("Initializing Abstract Repository with properties: {} ", properties);
        dbUtils = new JdbcUtils(properties);
        this.sqlCommand = "";
    }

    protected abstract T extractEntity(ResultSet resultSet) throws SQLException;

    protected abstract PreparedStatement createInsertStatement(Connection connection, T entity) throws SQLException;

    protected abstract PreparedStatement createUpdateStatement(Connection connection, T entity, ID id) throws SQLException;

    protected abstract PreparedStatement createDeleteStatement(Connection connection, T entity) throws SQLException;

    protected PreparedStatement createStatementFromEntity(Connection connection, T entity) throws SQLException {
        char command = sqlCommand.charAt(0);
        return switch (command) {
            case 'I' -> createInsertStatement(connection, entity);
            case 'D' -> createDeleteStatement(connection, entity);
            default -> null;
        };
    }

    private void executeCommand(T entity) {
        Connection connection = dbUtils.getConnection();
        try (PreparedStatement statement = createStatementFromEntity(connection, entity)
        ) {
            int result = statement.executeUpdate();
            logger.trace("Modified {} instances", result);
        } catch (SQLException e) {
            logger.error(e);
            e.printStackTrace();
        }
    }

    private void executeCommand(T entity, ID id) {
        Connection connection = dbUtils.getConnection();
        try (PreparedStatement statement = createUpdateStatement(connection, entity, id)
        ) {
            int result = statement.executeUpdate();
            logger.trace("Modified {} instances", result);
        } catch (SQLException e) {
            logger.error(e);
            e.printStackTrace();
        }
    }

    private Collection<T> executeQuery() {
        List<T> elements = new ArrayList<>();
        Connection connection = dbUtils.getConnection();
        try (PreparedStatement statement = connection.prepareStatement(sqlCommand);
             ResultSet resultSet = statement.executeQuery()
        ) {
            while (resultSet.next()) {
                elements.add(extractEntity(resultSet));
            }
        } catch (SQLException e) {
            logger.error(e);
            e.printStackTrace();
        }
        logger.traceExit();
        return elements;
    }

    @Override
    public void add(T elem) {
        logger.traceEntry("Saving entity {}", elem);
        executeCommand(elem);
        logger.traceExit();
    }

    @Override
    public void delete(T elem) {
        logger.traceEntry("Deleting entity {}", elem);
        executeCommand(elem);
        logger.traceExit();
    }

    @Override
    public void update(T elem, ID id) {
        logger.traceEntry("Updating entity {}", elem);
        executeCommand(elem, id);
        logger.traceExit();
    }

    public T getOne() {
        logger.traceEntry();
        Collection<T> result = executeQuery();
        if (result.isEmpty())
            return null;
        return executeQuery().iterator().next();
    }

    @Override
    public Collection<T> getAll() {
        logger.traceEntry();
        return executeQuery();
    }
}
