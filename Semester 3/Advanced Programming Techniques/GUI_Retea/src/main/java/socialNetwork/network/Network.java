package network;

import domain.Friendship;
import domain.User;
import exceptions.NetworkException;

import java.util.Vector;

public interface Network {
    /**
     * Adds an entity to network
     *
     * @param entity - entity to be added in network
     */
    void add(User entity);

    /**
     * Removes an entity to network
     *
     * @param entity - entity to be removed
     */
    void remove(User entity);

    /**
     * Makes a friendship relation between two entities
     *
     * @param friendship relatie de prietenie dintre 2 utilizatori
     * @throws NetworkException if they cannot be made friends
     */
    void makeFriends(Friendship friendship) throws NetworkException;

    /**
     * Removes a friendship between two entities
     *
     * @param friendship relatie de prietenie dintre 2 utilizatori
     * @throws NetworkException if they are not friends
     */
    void removeFriends(Friendship friendship) throws NetworkException;

    /**
     * @return how many communities are in the network
     */
    int getNumberOfCommunities();

    /**
     * @return the community with the most people
     */
    Vector<Long> getMostPopulatedCommunity();
}
