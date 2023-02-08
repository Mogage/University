package clinica.repository;

import clinica.domain.Sectie;

import java.sql.ResultSet;
import java.sql.SQLException;

public class SectiiRepository extends AbstractRepository<Long, Sectie> {


    public SectiiRepository(String url, String userName, String password) {
        super(url, userName, password, "SELECT * FROM sectii");
    }

    @Override
    protected Sectie extractEntity(ResultSet resultSet) throws SQLException {
        Long id = resultSet.getLong("id");
        Long idSefDeSectie = resultSet.getLong("id_sef_de_sectie");
        int pret = resultSet.getInt("pret_per_consultatie");
        int durata = resultSet.getInt("durata_maxima_consultatie");
        String nume = resultSet.getString("nume");
        Sectie sectie = new Sectie(nume, idSefDeSectie, pret, durata);
        sectie.setId(id);
        return sectie;
    }

    @Override
    public Iterable<Sectie> getAll() {
        super.setSqlCommand("SELECT * FROM " +
                "(SELECT id, nume, id_sef_de_sectie, pret_per_consultatie, durata_maxima_consultatie," +
                "        ROW_NUMBER() OVER (PARTITION BY nume ORDER BY id) AS row_number " +
                "FROM sectii GROUP BY sectii.nume, sectii.id) AS a WHERE a.row_number = 1");
        return super.getAll();
    }

    @Override
    public Sectie findAfterId(Long id) {
        super.setSqlCommand("SELECT * FROM sectii WHERE id=" + id.toString());
        return super.findAfterId(id);
    }
}
