package examen.repository;

import examen.domain.Nevoie;

import java.sql.*;
import java.time.LocalDateTime;

public class NevoiRepository extends AbstractRepository<Long, Nevoie> {

    public NevoiRepository(String url, String userName, String password) {
        super(url, userName, password);
    }

    @Override
    protected Nevoie extractEntity(ResultSet resultSet) throws SQLException {
        Long id = resultSet.getLong("id");
        String titlu = resultSet.getString("titlu");
        LocalDateTime deadline = LocalDateTime.parse(resultSet.getString("deadline"));
        Long omInNevoie = resultSet.getLong("om_in_nevoie");
        Long omSalvator = resultSet.getLong("om_salvator");
        String status = resultSet.getString("status");
        String descriere = resultSet.getString("descriere");
        Nevoie nevoie = new Nevoie(titlu, descriere, deadline, omInNevoie, omSalvator, status);
        nevoie.setId(id);
        return nevoie;
    }

    @Override
    protected PreparedStatement createStatementFromEntity(Connection connection, Nevoie entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.sqlCommand);
        statement.setLong(1, entity.getId());
        statement.setString(2, entity.getTitlu());
        statement.setString(3, entity.getDescriere());
        statement.setString(4, entity.getDeadline().toString());
        statement.setLong(5, entity.getOmInNevoie());
        statement.setString(6, entity.getStatus());
        return statement;
    }

    @Override
    public void save(Nevoie entity) throws Exception {
        super.sqlCommand = "INSERT INTO nevoi(id, titlu, descriere, deadline, om_in_nevoie, status) VALUES (?, ?, ?, ?, ?, ?)";
        super.save(entity);
    }

    @Override
    public Iterable<Nevoie> getAll() {
        super.sqlCommand = "SELECT * FROM nevoi";
        return super.getAll();
    }

    public void update(Nevoie nevoie) {
        super.sqlCommand = "UPDATE nevoi SET om_salvator=?, status=? WHERE id=?";
        try (Connection connection = DriverManager.getConnection(super.url, super.userName, super.password);
             PreparedStatement statement = connection.prepareStatement(super.sqlCommand)
        ) {
            statement.setLong(1, nevoie.getOmSalvator());
            statement.setString(2, nevoie.getStatus());
            statement.setLong(3, nevoie.getId());
            statement.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public Nevoie findAfterId(Long id) {
        super.sqlCommand = "SELECT * FROM nevoi WHERE id=" + id.toString();
        return super.findAfterId(id);
    }
}
