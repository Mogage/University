package com.socialNetwork.repository.databaseSystem;

import com.socialNetwork.domain.Friendship;
import com.socialNetwork.domain.User;
import com.socialNetwork.exceptions.RepositoryException;

import java.sql.*;
import java.util.Objects;

public class UserDBRepository extends AbstractDBRepository<Long, User> {

    public UserDBRepository(String url, String userName, String password) {
        super(url, userName, password, "SELECT * FROM users");
    }

    @Override
    protected User extractEntity(ResultSet resultSet) throws SQLException {
        Long id = resultSet.getLong("id_user");
        String firstName = resultSet.getString("first_name");
        String lastName = resultSet.getString("last_name");
        String email = resultSet.getString("email");
        String password = resultSet.getString("password");
        User user = new User(firstName, lastName, email, password);
        user.setId(id);
        return user;
    }

    @Override
    protected PreparedStatement createInsertStatement(Connection connection, User entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.getSqlCommand());
        statement.setLong(1, entity.getId());
        statement.setString(2, entity.getFirstName());
        statement.setString(3, entity.getLastName());
        statement.setString(4, entity.getEmail());
        statement.setString(5, entity.getPassword());
        return statement;
    }

    @Override
    protected PreparedStatement createUpdateStatement(Connection connection, User entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.getSqlCommand());
        statement.setString(1, entity.getFirstName());
        statement.setString(2, entity.getLastName());
        statement.setString(3, entity.getEmail());
        statement.setString(4, entity.getPassword());
        statement.setLong(5, entity.getId());
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
        super.setSqlCommand("INSERT INTO users (id_user, first_name, last_name, email, password) VALUES (?, ?, ?, ?, ?)");
        super.save(entity);
    }

    @Override
    public void update(Long aLong, User entity) throws IllegalArgumentException, RepositoryException {
        super.setSqlCommand("UPDATE users SET first_name=?, last_name=?, email=?, password=? WHERE id_user=?");
        super.update(aLong, entity);
    }

    @Override
    public User delete(Long aLong) throws RepositoryException {
        super.setSqlCommand("DELETE FROM Users WHERE id_user=?");
        return super.delete(aLong);
    }

    public User findAfterEmail(String email) throws RepositoryException{
        Iterable<User> users = super.getAll();

        for (User user : users) {
            if(Objects.equals(user.getEmail(), email)) {
                return user;
            }
        }

        throw new RepositoryException("User with this email does not exist.");
    }
}
