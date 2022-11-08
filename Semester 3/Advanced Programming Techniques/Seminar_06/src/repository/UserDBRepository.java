package repository;

import domain.User;

import java.sql.*;
import java.util.HashSet;
import java.util.Set;

public class UserDBRepository implements Repository<Long, User> {
    private String url;
    private String userName;
    private String pass;

    public UserDBRepository(String url, String userName, String pass) {
        this.url = url;
        this.userName = userName;
        this.pass = pass;
    }

    @Override
    public User findOne(Long aLong) {
        return null;
    }

    @Override
    public Iterable<User> findAll() {
        Set<User> userSet = new HashSet<>();
        try (Connection connection = DriverManager.getConnection(url, userName, pass);
             PreparedStatement statement = connection.prepareStatement("SELECT * FROM users");
             ResultSet resultSet = statement.executeQuery();
        ) {
            while (resultSet.next()) {
                Long id = resultSet.getLong("id");
                String firstName = resultSet.getString("first_name");
                String lastName = resultSet.getString("last_name");
                User newUser = new User(lastName, firstName);
                newUser.setId(id);
                userSet.add(newUser);
            }
        } catch (SQLException exception) {
            exception.printStackTrace();
        }
        return userSet;
    }

    @Override
    public User save(User entity) {
        String sql = "INSERT INTO users (first_name, last_name) VALUES (?, ?)";
        try (Connection connection = DriverManager.getConnection(url, userName, pass);
             PreparedStatement statement = connection.prepareStatement(sql)
        ) {
            statement.setString(1, entity.getFirstName());
            statement.setString(2, entity.getLastName());
            statement.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public User delete(Long aLong) {
        return null;
    }

    @Override
    public User update(User Entity) {
        return null;
    }
}
;