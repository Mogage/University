package com.socialNetwork.repository;

import com.socialNetwork.domain.Entity;
import com.socialNetwork.exceptions.RepositoryException;

public interface Repository<ID, T extends Entity<ID>> {
    /**
     * @param obj entity to be added
     * @throws RepositoryException if the entity already exists
     */
    void save(T obj) throws RepositoryException;

    /**
     * @param id  id of the object to modify
     * @param obj the new values of the object
     * @throws RepositoryException if the id does not exists
     */
    void update(ID id, T obj) throws RepositoryException;

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

    /**
     * @return number of entities from repo
     */
    int size();
}
