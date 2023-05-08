package root.proiect_mpp.repositories.people;

import root.proiect_mpp.domain.people.Person;
import root.proiect_mpp.repositories.Repository;

import java.util.Collection;

public interface IPersonRepository<T extends Person> extends Repository<T, Integer> {
    Collection<T> getPersonByFirstName(String firstName);
    Collection<T> getPersonByLastName(String lastName);
}
