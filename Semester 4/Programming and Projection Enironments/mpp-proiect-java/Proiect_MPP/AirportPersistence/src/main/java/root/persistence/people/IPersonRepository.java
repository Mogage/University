package root.persistence.people;

import root.model.people.Person;
import root.persistence.Repository;

import java.util.Collection;

public interface IPersonRepository<T extends Person> extends Repository<T, Integer> {
    Collection<T> getPersonByFirstName(String firstName);
    Collection<T> getPersonByLastName(String lastName);
}
