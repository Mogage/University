package repository;

import model.validators.ValidationException;

public interface Repository<ID, E> {
    E save(E entity) throws ValidationException;
    E delete(ID id);
    E findOne(ID id);
    Iterable<E> findAll();
}
