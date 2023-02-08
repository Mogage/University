package anar.repository;

import anar.domain.Rau;

import java.sql.*;

public class RauriRepository extends AbstractRepository<Rau> {

    public RauriRepository(String url, String userName, String password) {
        super(url, userName, password);
    }

    @Override
    protected Rau extractEntity(ResultSet resultSet) throws SQLException {
        String nume = resultSet.getString("nume");
        Integer cota = resultSet.getInt("cota_medie");
        return new Rau(nume, cota);
    }

    @Override
    public Iterable<Rau> getAll() {
        super.sqlCommand = "SELECT * FROM rauri";
        return super.getAll();
    }

    public void update(Rau rau, Integer cota) {
        super.sqlCommand = "UPDATE rauri SET cota_medie=? WHERE nume=?";
        try (Connection connection = DriverManager.getConnection(super.url, super.userName, super.password);
             PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        ) {
            statement.setInt(1, cota);
            statement.setString(2, rau.getNume());
            statement.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public Rau findAfterName(String numeRau) {
        super.sqlCommand = "SELECT * FROM rauri WHERE nume=?";
        try (Connection connection = DriverManager.getConnection(super.url, super.userName, super.password);
             PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        ) {
            statement.setString(1, numeRau);
            ResultSet resultSet = statement.executeQuery();
            if (resultSet.next()) {
                return extractEntity(resultSet);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return null;
    }
}
