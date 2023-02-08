package ati.repository;

import ati.domain.Pat;

import java.sql.*;

public class PaturiRepository extends AbstractRepository<Pat> {

    public PaturiRepository(String url, String userName, String password) {
        super(url, userName, password);
    }

    @Override
    protected Pat extractEntity(ResultSet resultSet) throws SQLException {
        Long id = resultSet.getLong("id");
        String tip = resultSet.getString("tip");
        boolean ventilatie = resultSet.getBoolean("ventilatie");
        String cnp = resultSet.getString("pacient");
        return new Pat(id, tip, ventilatie, cnp);
    }

    @Override
    public Iterable<Pat> getAll() {
        super.sqlCommand = "SELECT * FROM paturi";
        return super.getAll();
    }

    public boolean estePacientInternat(String cnp) {
        super.sqlCommand = "SELECT * FROM paturi WHERE paturi.pacient=?";
        try (Connection connection = DriverManager.getConnection(super.url, super.userName, super.password);
             PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        ) {
            statement.setString(1, cnp);
            ResultSet resultSet = statement.executeQuery();
            if (resultSet.next()) {
                return true;
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return false;
    }

    public Pat getPatLiber(String tipPat) {
        super.sqlCommand = "SELECT * FROM paturi WHERE paturi.tip=? AND paturi.pacient IS NULL";
        return getPat(tipPat);
    }

    public void updatePat(Pat pat) {
        super.sqlCommand = "UPDATE paturi SET pacient=? WHERE id=?";
        try (Connection connection = DriverManager.getConnection(super.url, super.userName, super.password);
             PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        ) {
            statement.setString(1, pat.getCnpPacient());
            statement.setLong(2, pat.getId());
            statement.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public Pat getPacientPat(String cnp) {
        super.sqlCommand = "SELECT * FROM paturi WHERE paturi.pacient=?";
        return getPat(cnp);
    }

    private Pat getPat(String string) {
        try (Connection connection = DriverManager.getConnection(super.url, super.userName, super.password);
             PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        ) {
            statement.setString(1, string);
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
