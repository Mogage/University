package com.socialNetwork.service;

import com.socialNetwork.domain.Entity;
import com.socialNetwork.domain.Friendship;
import com.socialNetwork.domain.User;
import com.socialNetwork.domain.validators.FriendshipValidator;
import com.socialNetwork.domain.validators.UserValidator;
import com.socialNetwork.domain.validators.Validator;
import com.socialNetwork.exceptions.NetworkException;
import com.socialNetwork.exceptions.RepositoryException;
import com.socialNetwork.exceptions.ValidationException;
import com.socialNetwork.network.Network;
import com.socialNetwork.network.MainNetwork;
import com.socialNetwork.repository.databaseSystem.FriendshipDBRepository;
import com.socialNetwork.repository.databaseSystem.UserDBRepository;
import com.socialNetwork.repository.Repository;


import java.time.LocalDateTime;
import java.util.Set;
import java.util.TreeSet;
import java.util.Vector;

public class MainService implements Service {

    private final UserValidator userValidator;
    private final FriendshipValidator friendshipValidator;
    private final UserDBRepository userRepository;
    private final FriendshipDBRepository friendshipRepository;
    private final MainNetwork network;

    public MainService(
            Validator<User> validator,
            Validator<Friendship> friendshipValidator,
            Repository<Long, User> userRepository,
            Repository<Long, Friendship> friendshipRepository,
            Network network) {
        this.userValidator = (UserValidator) validator;
        this.friendshipValidator = (FriendshipValidator) friendshipValidator;
        this.userRepository = (UserDBRepository) userRepository;
        this.friendshipRepository = (FriendshipDBRepository) friendshipRepository;
        this.network = (MainNetwork) network;

        Iterable<User> users = userRepository.getAll();
        for (User user : users) {
            network.add(user);
        }

        Iterable<Friendship> friendships = friendshipRepository.getAll();
        for (Friendship friendship : friendships) {
            try {
                network.makeFriends(friendship);
            } catch (NetworkException e) {
                e.printStackTrace();
            }
        }
    }

    private Long getId(Iterable<? extends Entity<Long>> entities) {
        Set<Long> distinct = new TreeSet<>();
        long id = 1L;

        for (Entity<Long> entity : entities) {
            distinct.add(entity.getId());
        }

        while (true) {
            if (!distinct.contains(id)) {
                return id;
            }
            id = id + 1;
        }
    }

    /**
     * /@todo: implement hash function
     *
     * @param password string to hash
     * @return a string with the hashed password
     */
    private String hash(String password) {
        return password;
    }

    @Override
    public void add(String firstName, String lastName, String email, String password) throws ValidationException, RepositoryException {
        Long id = getId(userRepository.getAll());
        User toAdd = new User(firstName, lastName, email, hash(password));
        toAdd.setId(id);
        userValidator.validate(toAdd);
        network.add(toAdd);
        userRepository.save(toAdd);
    }

    @Override
    public void updateUser(Long id, String firstName, String lastName, String email, String password) throws RepositoryException, ValidationException {
        User newUser = new User(firstName, lastName, email, hash(password));
        newUser.setId(id);
        userValidator.validate(newUser);
        User oldUser = userRepository.findAfterId(id);
        network.remove(oldUser);
        network.add(newUser);
        userRepository.update(id, newUser);
    }

    @Override
    public User remove(Long id) throws RepositoryException, NetworkException {
        User toDelete = userRepository.findAfterId(id);
        Vector<Friendship> userFriendships = friendshipRepository.findUserFriends(id);
        for (Friendship friendship : userFriendships) {
            network.removeFriends(friendship);
            friendshipRepository.delete(friendship.getId());
        }
        network.remove(toDelete);
        return userRepository.delete(id);
    }

    @Override
    public User getUser(Long id) throws RepositoryException {
        return userRepository.findAfterId(id);
    }

    @Override
    public User findUserAfterEmail(String email) throws RepositoryException {
        return userRepository.findAfterEmail(email);
    }

    @Override
    public void makeFriends(Long id1, Long id2) throws NetworkException, ValidationException, RepositoryException {
        userRepository.findAfterId(id1);
        userRepository.findAfterId(id2);
        Long id = getId(friendshipRepository.getAll());
        Friendship friendship = new Friendship(id1, id2, LocalDateTime.now());
        friendship.setId(id);
        friendshipValidator.validate(friendship);
        network.makeFriends(friendship);
        friendshipRepository.save(friendship);
    }

    @Override
    public void updateFriends(Long friendshipId, Long idUser1, Long idUser2) throws ValidationException, RepositoryException, NetworkException {
        Friendship oldFriendship = friendshipRepository.findAfterId(friendshipId);
        Friendship newFriendship = new Friendship(idUser1, idUser2, oldFriendship.getFriendsFrom());
        newFriendship.setId(friendshipId);
        friendshipValidator.validate(newFriendship);
        network.removeFriends(oldFriendship);
        network.makeFriends(newFriendship);
        friendshipRepository.update(friendshipId, newFriendship);
    }

    @Override
    public void removeFriends(Long id) throws NetworkException, RepositoryException {
        Friendship friendship = friendshipRepository.findAfterId(id);
        network.removeFriends(friendship);
        friendshipRepository.delete(friendship.getId());
    }

    @Override
    public int numberOfCommunities() {
        return network.getNumberOfCommunities();
    }

    @Override
    public Vector<User> mostPopulatedCommunity() {
        Vector<Long> communityIds = network.getMostPopulatedCommunity();
        Vector<User> community = new Vector<>();
        for (Long id : communityIds) {
            try {
                community.add(userRepository.findAfterId(id));
            } catch (RepositoryException e) {
                System.out.println(e.getMessage());
            }
        }
        return community;
    }

    @Override
    public Iterable<User> getAllUsers() {
        return userRepository.getAll();
    }

    @Override
    public Iterable<Friendship> getAllFriendships() {
        return friendshipRepository.getAll();
    }

    @Override
    public int numberOfUsers() {
        return userRepository.size();
    }

    @Override
    public int numberOfFriendships() {
        return friendshipRepository.size();
    }
}
