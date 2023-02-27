package examen.repository;

import examen.domain.Persoana;

import java.sql.*;

public class PersoaneRepository extends AbstractRepository<Long, Persoana> {
    public PersoaneRepository(String url, String userName, String password) {
        super(url, userName, password);
    }

    @Override
    protected Persoana extractEntity(ResultSet resultSet) throws SQLException {
        String nume = resultSet.getString("nume");
        String prenume = resultSet.getString("prenume");
        String username = resultSet.getString("username");
        String parola = resultSet.getString("parola");
        String oras = resultSet.getString("oras");
        String strada = resultSet.getString("strada");
        String numarStrada = resultSet.getString("numar_strada");
        String telefon = resultSet.getString("telefon");
        Long id = resultSet.getLong("id");
        Persoana persoana = new Persoana(nume, prenume, username, parola, oras, strada, numarStrada, telefon);
        persoana.setId(id);
        return persoana;
    }

    @Override
    protected PreparedStatement createStatementFromEntity(Connection connection, Persoana entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        statement.setLong(1, entity.getId());
        statement.setString(2, entity.getNume());
        statement.setString(3, entity.getPrenume());
        statement.setString(4, entity.getUsername());
        statement.setString(5, entity.getParola());
        statement.setString(6, entity.getOras());
        statement.setString(7, entity.getStrada());
        statement.setString(8, entity.getNumarStrada());
        statement.setString(9, entity.getTelefon());
        return statement;
    }

    @Override
    public Iterable<Persoana> getAll() {
        super.sqlCommand = "SELECT * FROM persoane";
        return super.getAll();
    }

    @Override
    public void save(Persoana entity) throws Exception {
        super.sqlCommand = "INSERT INTO persoane VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";
        super.save(entity);
    }

    public Persoana findAfterId(Long omInNevoie) {
        super.sqlCommand = "SELECT * FROM persoane WHERE id=" + omInNevoie.toString();
        return super.findAfterId(omInNevoie);
    }
}
