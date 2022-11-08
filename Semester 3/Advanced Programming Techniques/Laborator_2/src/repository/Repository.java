package repository;

import domain.Entity;
import exceptions.RepositoryException;

import java.util.Vector;

public interface Repository<ID, T extends Entity<ID>> {
    /**
     * @param obj entity to be added
     * @throws RepositoryException if the entity already exists
     */
    void save(T obj) throws RepositoryException;

    /**
     * @param id id of the entity to be deleted
     * @return entity that was deleted
     * @throws RepositoryException if the entity does not exist
     */
    T delete(ID id) throws RepositoryException;

    /**
     * @param id id of the entity to look after
     * @return entity with that id
     * @throws RepositoryException if entity does not exist
     */
    T findAfterId(ID id) throws RepositoryException;

    /**
     * @return iterable with all entities
     */
    Iterable<T> getAll();
}
