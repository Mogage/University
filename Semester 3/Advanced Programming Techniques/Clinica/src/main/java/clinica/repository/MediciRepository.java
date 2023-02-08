package clinica.repository;

import clinica.domain.Medic;

import java.sql.ResultSet;
import java.sql.SQLException;

public class MediciRepository  extends AbstractRepository<Long, Medic> {

    public MediciRepository(String url, String userName, String password) {
        super(url, userName, password, "SELECT * FROM medici");
    }

    @Override
    protected Medic extractEntity(ResultSet resultSet) throws SQLException {
        Long id = resultSet.getLong("id");
        Long idSectie = resultSet.getLong("id_sectie");
        int vechime = resultSet.getInt("vechime");
        boolean rezident = resultSet.getBoolean("rezident");
        String nume = resultSet.getString("nume");
        Medic medic = new Medic(idSectie, nume, vechime, rezident);
        medic.setId(id);
        return medic;
    }

    @Override
    public Iterable<Medic> getAll() {
        super.setSqlCommand("SELECT * FROM medici");
        return super.getAll();
    }

    @Override
    public Medic findAfterId(Long id) {
        super.setSqlCommand("SELECT * FROM medici WHERE id=" + id.toString());
        return super.findAfterId(id);
    }
}