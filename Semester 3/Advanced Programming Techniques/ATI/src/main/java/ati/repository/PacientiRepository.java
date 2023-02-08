package ati.repository;

import ati.domain.Pacient;

import java.sql.*;

public class PacientiRepository extends AbstractRepository<Pacient> {

    public PacientiRepository(String url, String userName, String password) {
        super(url, userName, password);
    }

    @Override
    protected Pacient extractEntity(ResultSet resultSet) throws SQLException {
        String cnp = resultSet.getString("cnp");
        int varsta = resultSet.getInt("varsta");
        boolean prematur = resultSet.getBoolean("prematur");
        String diagnostic = resultSet.getString("diagnostic");
        int gravitate = resultSet.getInt("gravitate");
        return new Pacient(cnp, varsta, prematur, diagnostic, gravitate);
    }

    @Override
    public Iterable<Pacient> getAll() {
        super.sqlCommand = "SELECT * FROM pacienti";
        return super.getAll();
    }

    public void sterge(String cnp) {
        super.sqlCommand = "DELETE FROM pacienti WHERE pacienti.cnp=?";
        try (Connection connection = DriverManager.getConnection(super.url, super.userName, super.password);
             PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        ) {
            statement.setString(1, cnp);
            statement.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
