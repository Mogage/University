package root.proiect_mpp.repositories.people;

import root.proiect_mpp.domain.people.Person;
import root.proiect_mpp.repositories.AbstractRepository;

import java.util.Collection;
import java.util.Properties;

public abstract class PersonRepository<T extends Person> extends AbstractRepository<T, Integer> implements IPersonRepository<T> {
    public PersonRepository(Properties properties) {
        super(properties);
        logger.info("Initializing Person Repository ");
    }

    protected abstract String getTableName();

    @Override
    public Collection<T> getPersonByFirstName(String firstName) {
        super.sqlCommand = "SELECT * FROM " + getTableName() + " WHERE first_name='" + firstName + '\'';
        return super.getAll();
    }

    @Override
    public Collection<T> getPersonByLastName(String lastName) {
        super.sqlCommand = "SELECT * FROM " + getTableName() + " WHERE last_name='" + lastName + '\'';
        return super.getAll();
    }
}
