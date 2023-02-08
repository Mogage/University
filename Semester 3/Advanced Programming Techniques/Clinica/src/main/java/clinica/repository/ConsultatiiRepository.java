package clinica.repository;

import clinica.domain.Consultatie;
import clinica.utils.Observable;

import java.sql.*;
import java.time.LocalDate;
import java.time.LocalTime;

public class ConsultatiiRepository extends AbstractRepository<Long, Consultatie> {

    public ConsultatiiRepository(String url, String userName, String password) {
        super(url, userName, password, "SELECT * FROM consultatii");
    }

    @Override
    protected Consultatie extractEntity(ResultSet resultSet) throws SQLException {
        Long id = resultSet.getLong("id");
        Long idMedic = resultSet.getLong("id_medic");
        String cnpPacient = resultSet.getString("cnp_pacient");
        String numePacient = resultSet.getString("nume_pacient");
        LocalDate data = LocalDate.parse(resultSet.getString("data"));
        LocalTime ora = LocalTime.parse(resultSet.getString("ora"));
        Consultatie consultatie = new Consultatie(idMedic, numePacient, cnpPacient, data, ora);
        consultatie.setId(id);
        return consultatie;
    }

    @Override
    public Iterable<Consultatie> getAll() {
        super.setSqlCommand("SELECT * FROM consultatii");
        return super.getAll();
    }

    @Override
    public Consultatie findAfterId(Long id) {
        super.setSqlCommand("SELECT * FROM consultatii WHERE id=" + id.toString());
        return super.findAfterId(id);
    }

    protected PreparedStatement createStatement(Connection connection, Consultatie entity) throws SQLException {
        super.setSqlCommand("INSERT INTO consultatii(id, id_medic, cnp_pacient, nume_pacient, data, ora) VALUES(?, ?, ?, ?, ?, ?)");
        PreparedStatement statement = connection.prepareStatement(super.getSqlCommand());
        statement.setLong(1, entity.getId());
        statement.setLong(2, entity.getIdMedic());
        statement.setString(3, entity.getCnpPacient());
        statement.setString(4, entity.getNumePacient());
        statement.setString(5, entity.getData().toString());
        statement.setString(6, entity.getOra().toString());
        return statement;
    }

    public void add(Consultatie entity) {
        try (Connection connection = DriverManager.getConnection(super.url, super.userName, super.password);
             PreparedStatement statement = createStatement(connection, entity);
        ) {
            statement.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public void delete(Long id) {
        super.setSqlCommand("DELETE FROM consultatii WHERE id=" + id.toString());
        try (Connection connection = DriverManager.getConnection(super.url, super.userName, super.password);
             PreparedStatement statement = connection.prepareStatement(super.getSqlCommand());
        ) {
            statement.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
