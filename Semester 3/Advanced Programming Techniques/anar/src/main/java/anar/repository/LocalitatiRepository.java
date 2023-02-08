package anar.repository;

import anar.domain.Localitate;
import anar.domain.Rau;

import java.sql.*;

public class LocalitatiRepository extends AbstractRepository<Localitate> {

    public LocalitatiRepository(String url, String userName, String password) {
        super(url, userName, password);
    }

    @Override
    protected Localitate extractEntity(ResultSet resultSet) throws SQLException {
        String nume = resultSet.getString("nume");
        String numeRau = resultSet.getString("nume_rau");
        Integer cotaRau = resultSet.getInt("cota_rau");
        Integer cotaMinima = resultSet.getInt("cota_minima_de_risc");
        Integer cotaMaxima = resultSet.getInt("cota_maxima_admisa");
        return new Localitate(nume, new Rau(numeRau, cotaRau), cotaMinima, cotaMaxima);
    }

    @Override
    public Iterable<Localitate> getAll() {
        super.sqlCommand = "SELECT * FROM localitati";
        return super.getAll();
    }

    public void update(Localitate localitate, Integer cota) {
        super.sqlCommand = "UPDATE localitati SET cota_rau=? WHERE nume=?";
        try (Connection connection = DriverManager.getConnection(super.url, super.userName, super.password);
             PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        ) {
            statement.setInt(1, cota);
            statement.setString(2, localitate.getNume());
            statement.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
