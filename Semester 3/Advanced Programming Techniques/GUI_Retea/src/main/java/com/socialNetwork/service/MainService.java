package com.socialNetwork.service;

import com.socialNetwork.domain.Entity;
import com.socialNetwork.domain.Friendship;
import com.socialNetwork.domain.Message;
import com.socialNetwork.domain.User;
import com.socialNetwork.domain.validators.FriendshipValidator;
import com.socialNetwork.domain.validators.MessageValidator;
import com.socialNetwork.domain.validators.UserValidator;
import com.socialNetwork.domain.validators.Validator;
import com.socialNetwork.exceptions.NetworkException;
import com.socialNetwork.exceptions.RepositoryException;
import com.socialNetwork.exceptions.ValidationException;
import com.socialNetwork.network.Network;
import com.socialNetwork.network.MainNetwork;
import com.socialNetwork.repository.databaseSystem.FriendshipDBRepository;
import com.socialNetwork.repository.databaseSystem.MessagesDBRepository;
import com.socialNetwork.repository.databaseSystem.UserDBRepository;
import com.socialNetwork.repository.Repository;


import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

public class MainService implements Service {

    private final UserValidator userValidator;
    private final FriendshipValidator friendshipValidator;
    private final MessageValidator messageValidator;
    private final UserDBRepository userRepository;
    private final FriendshipDBRepository friendshipRepository;
    private final MessagesDBRepository messagesRepository;
    private final MainNetwork network;

    public MainService(
            Validator<User> validator,
            Validator<Friendship> friendshipValidator,
            Validator<Message> messageValidator,
            Repository<Long, User> userRepository,
            Repository<Long, Friendship> friendshipRepository,
            Repository<Long, Message> messageRepository,
            Network network) {
        this.userValidator = (UserValidator) validator;
        this.friendshipValidator = (FriendshipValidator) friendshipValidator;
        this.messageValidator = (MessageValidator) messageValidator;
        this.userRepository = (UserDBRepository) userRepository;
        this.friendshipRepository = (FriendshipDBRepository) friendshipRepository;
        this.messagesRepository = (MessagesDBRepository) messageRepository;
        this.network = (MainNetwork) network;

        createNetwork(this.userRepository, this.friendshipRepository);
    }

    private void createNetwork(UserDBRepository userRepository, FriendshipDBRepository friendshipRepository) {
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
        List<Friendship> userFriendships = friendshipRepository.findUserFriends(id);
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
    public Friendship getFriendship(Long id) throws RepositoryException {
        return friendshipRepository.findAfterId(id);
    }

    @Override
    public User findUserAfterEmail(String email) throws RepositoryException {
        return userRepository.findAfterEmail(email);
    }

    private List<Friendship> getFriendships(Long id, String friendshipType) {
        return friendshipRepository.findUserFriends(id).stream()
                .filter(friendship -> Objects.equals(friendship.getStatus(), friendshipType))
                .collect(Collectors.toList());
    }

    @Override
    public List<Friendship> findUserFriends(Long id) {
        return getFriendships(id, "Friends");
    }

    @Override
    public List<Friendship> findUserRequests(Long id) {
        return getFriendships(id, "Pending");
    }

    @Override
    public void makeFriends(Long id1, Long id2) throws NetworkException, ValidationException, RepositoryException {
        userRepository.findAfterId(id1);
        userRepository.findAfterId(id2);
        Long id = getId(friendshipRepository.getAll());
        Friendship friendship = new Friendship(id1, id2, LocalDateTime.now(), "Pending");
        friendship.setId(id);
        friendshipValidator.validate(friendship);
        network.makeFriends(friendship);
        friendshipRepository.save(friendship);
    }

    @Override
    public void updateFriends(Long friendshipId, Long idUser1, Long idUser2) throws ValidationException, RepositoryException, NetworkException {
        Friendship oldFriendship = friendshipRepository.findAfterId(friendshipId);
        Friendship newFriendship = new Friendship(idUser1, idUser2, LocalDateTime.now(), "Friends");
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
    public List<User> mostPopulatedCommunity() {
        List<Long> communityIds = network.getMostPopulatedCommunity();
        List<User> community = new ArrayList<>();
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

    @Override
    public void refreshConversation() {
        messagesRepository.clearData();
        messagesRepository.loadData("SELECT * FROM messages");
    }

    @Override
    public void refresh() {
        userRepository.clearData();
        userRepository.loadData("SELECT * FROM users");
        friendshipRepository.clearData();
        friendshipRepository.loadData("SELECT * FROM friendships");
        createNetwork(userRepository, friendshipRepository);
        refreshConversation();
    }

    @Override
    public List<Message> getMessages(Long friendshipId) {
        List<Message> friendsMessages = messagesRepository.getAll();
        return friendsMessages.stream().filter(message -> Objects.equals(message.getFriendshipId(), friendshipId)).collect(Collectors.toList());
    }

    @Override
    public void sendMessage(Long friendshipId, String text, Long senderId, Long receiverId) throws ValidationException, RepositoryException {
        Message message = new Message(text, senderId, receiverId, friendshipId);
        message.setId(getId(messagesRepository.getAll()));
        messageValidator.validate(message);
        messagesRepository.save(message);
    }
}
