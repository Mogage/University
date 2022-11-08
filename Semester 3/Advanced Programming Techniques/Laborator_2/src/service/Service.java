package service;

import domain.User;
import exceptions.NetworkException;
import exceptions.RepositoryException;
import exceptions.ValidationException;

import java.util.Vector;

public interface Service<T> {
    /**
     * @param firstName  = name of the entity
     * @param lastName - name of the entity
     * @throws ValidationException if the entity is not valid
     * @throws RepositoryException if the entity already exists
     */
    void add(String firstName, String lastName) throws ValidationException, RepositoryException;

    /**
     * @param id id of the entity to remove
     * @return the entity that was removed
     * @throws RepositoryException if the entity does not exists
     */
    T remove(long id) throws RepositoryException;

    /**
     * @param id1 id of the first entity to add friendship
     * @param id2 id of the second entity to add friendship
     * @throws NetworkException if they are already friends
     */
    void makeFriends(long id1, long id2) throws NetworkException, ValidationException;

    /**
     * @param id1 id of the first entity to remove the friendship
     * @param id2 id of the second entity to remove the friendship
     * @throws NetworkException if they are not friends
     */
    void removeFriends(long id1, long id2) throws NetworkException, ValidationException;

    /**
     * @return number of communities
     */
    int numberOfCommunities();

    /**
     * @return the community with the most people
     */
    Vector<User> mostPopulatedCommunity();

    /**
     * @return get all users
     */
    Iterable<T> getAll();
}
