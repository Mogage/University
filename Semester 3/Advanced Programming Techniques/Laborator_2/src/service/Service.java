package service;

import domain.Friendship;
import domain.User;
import exceptions.NetworkException;
import exceptions.RepositoryException;
import exceptions.ValidationException;

import java.util.Vector;

public interface Service {
    /**
     * @param firstName = name of the entity
     * @param lastName  - name of the entity
     * @throws ValidationException if the entity is not valid
     * @throws RepositoryException if the entity already exists
     */
    void add(String firstName, String lastName) throws ValidationException, RepositoryException;

    void updateUser(long id, String firstName, String lastName) throws RepositoryException, ValidationException;

    /**
     * @param id id of the entity to remove
     * @return the entity that was removed
     * @throws RepositoryException if the entity does not exist
     */
    User remove(long id) throws RepositoryException, NetworkException;

    /**
     * @param id1 id of the first entity to add friendship
     * @param id2 id of the second entity to add friendship
     * @throws NetworkException if they are already friends
     */
    void makeFriends(long id1, long id2) throws NetworkException, ValidationException, RepositoryException;

    void updateFriends(long friendshipId, long idUser1, long idUser2) throws ValidationException, RepositoryException, NetworkException;

    /**
     * @param id of the friendship to remove
     * @throws NetworkException if this friendship does not exist
     */
    void removeFriends(long id) throws NetworkException, ValidationException, RepositoryException;

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
    Iterable<User> getAllUsers();

    Iterable<Friendship> getAllFriendships();

    int numberOfUsers();

    int numberOfFriendships();
}
