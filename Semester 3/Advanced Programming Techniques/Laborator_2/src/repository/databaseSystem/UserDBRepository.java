package repository.databaseSystem;

import domain.User;
import exceptions.RepositoryException;

import java.sql.*;

public class UserDBRepository extends AbstractDBRepository<Long, User> {

    public UserDBRepository(String url, String userName, String password) {
        super(url, userName, password, "SELECT * FROM users");
    }

    @Override
    protected User extractEntity(ResultSet resultSet) throws SQLException {
        Long id = resultSet.getLong("id_user");
        String firstName = resultSet.getString("first_name");
        String lastName = resultSet.getString("last_name");
        User user = new User(firstName, lastName);
        user.setId(id);
        return user;
    }

    @Override
    protected PreparedStatement createInsertStatement(Connection connection, User entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.getSqlCommand());
        statement.setLong(1, entity.getId());
        statement.setString(2, entity.getFirstName());
        statement.setString(3, entity.getLastName());
        return statement;
    }

    @Override
    protected PreparedStatement createUpdateStatement(Connection connection, User entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.getSqlCommand());
        statement.setString(1, entity.getFirstName());
        statement.setString(2, entity.getLastName());
        statement.setLong(3, entity.getId());
        return statement;
    }

    @Override
    protected PreparedStatement createDeleteStatement(Connection connection, User entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.getSqlCommand());
        statement.setLong(1, entity.getId());
        return statement;
    }

    @Override
    public void save(User entity) throws IllegalArgumentException, RepositoryException {
        super.setSqlCommand("INSERT INTO users (id_user, first_name, last_name) VALUES (?, ?, ?)");
        super.save(entity);
    }

    @Override
    public void update(Long aLong, User entity) throws IllegalArgumentException, RepositoryException {
        super.setSqlCommand("UPDATE users SET first_name=?, last_name=? WHERE id_user=?");
        super.update(aLong, entity);
    }

    @Override
    public User delete(Long aLong) throws RepositoryException {
        super.setSqlCommand("DELETE FROM Users WHERE id_user=?");
        return super.delete(aLong);
    }
}
/*
public class UserDBRepository extends InMemoryRepository<Long, User> {
    private final String url;
    private final String userName;
    private final String password;

    private String sqlCommand;

    public UserDBRepository(String url, String userName, String password) {
        super();
        this.url = url;
        this.userName = userName;
        this.password = password;
        loadData();
    }

    private void loadData() {
        this.sqlCommand = "SELECT * FROM users";

        try (Connection connection = DriverManager.getConnection(url, userName, password);
             PreparedStatement statement = connection.prepareStatement(sqlCommand);
             ResultSet resultSet = statement.executeQuery()
        ) {
            while (resultSet.next()) {
                Long id = resultSet.getLong("id_user");
                String firstName = resultSet.getString("first_name");
                String lastName = resultSet.getString("last_name");
                User newUser = new User(firstName, lastName);
                newUser.setId(id);
                super.save(newUser);
            }
        } catch (SQLException | RepositoryException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void save(User obj) throws RepositoryException {
        this.sqlCommand = "INSERT INTO users (id_user, first_name, last_name) VALUES (?, ?, ?)";
        try (Connection connection = DriverManager.getConnection(url, userName, password);
             PreparedStatement statement = connection.prepareStatement(sqlCommand)
        ) {
            statement.setLong(1, obj.getId());
            statement.setString(2, obj.getFirstName());
            statement.setString(3, obj.getLastName());
            statement.executeUpdate();
            super.save(obj);
        } catch (SQLException e) {
            throw new RepositoryException(e.getMessage());
        }
    }

    @Override
    public void update(Long aLong, User obj) throws RepositoryException {
        this.sqlCommand = "UPDATE users SET first_name=?, last_name=? WHERE id_user=?";
        try (Connection connection = DriverManager.getConnection(url, userName, password);
             PreparedStatement statement = connection.prepareStatement(sqlCommand)
        ) {
            statement.setString(1, obj.getFirstName());
            statement.setString(2, obj.getLastName());
            statement.setLong(3, obj.getId());
            statement.executeUpdate();
        } catch (SQLException e) {
            throw new RepositoryException(e.getMessage());
        }
        super.update(aLong, obj);
    }

    @Override
    public User delete(Long aLong) throws RepositoryException {
        this.sqlCommand = "DELETE FROM Users WHERE id_user=?";
        try (Connection connection = DriverManager.getConnection(url, userName, password);
             PreparedStatement statement = connection.prepareStatement(sqlCommand)
        ) {
            statement.setLong(1, aLong);
            statement.executeUpdate();
            return super.delete(aLong);
        } catch (SQLException e) {
            throw new RepositoryException(e.getMessage());
        }
    }
}
*/