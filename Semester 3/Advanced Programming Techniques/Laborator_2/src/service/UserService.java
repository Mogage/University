package service;

import domain.Friendship;
import domain.User;
import domain.validators.Validator;
import exceptions.NetworkException;
import exceptions.RepositoryException;
import exceptions.ValidationException;
import network.Network;
import repository.Repository;

import java.util.Set;
import java.util.TreeSet;
import java.util.Vector;

public class UserService implements Service<User> {

    private final Validator<User> userValidator;

    private final Validator<Friendship> friendshipValidator;
    private final Repository<Long, User> repository;

    private final Network<User> network;

    public UserService(Validator<User> validator, Validator<Friendship> friendshipValidator, Repository<Long, User> repository, Network<User> network) {
        this.userValidator = validator;
        this.friendshipValidator = friendshipValidator;
        this.repository = repository;
        this.network = network;

        User toAdd1 = new User(1, "nicu", "mog");
        User toAdd2 = new User(2, "teo", "sud");
        User toAdd3 = new User(3, "fabi", "martin");
        User toAdd4 = new User(4, "denis", "moldovan");
        User toAdd5 = new User(5, "daria", "ev");
        User toAdd6 = new User(6, "dragos", "moro");
        User toAdd7 = new User(7, "monty", "martin");
        try {
            this.repository.save(toAdd1);
            this.repository.save(toAdd2);
            this.repository.save(toAdd3);
            this.repository.save(toAdd4);
            this.repository.save(toAdd5);
            this.repository.save(toAdd6);
            this.repository.save(toAdd7);
            this.network.add(toAdd1);
            this.network.add(toAdd2);
            this.network.add(toAdd3);
            this.network.add(toAdd4);
            this.network.add(toAdd5);
            this.network.add(toAdd6);
            this.network.add(toAdd7);
        } catch (RepositoryException e) {
            throw new RuntimeException(e);
        }
    }

    private Long getId(Iterable<User> users) {
        Set<Long> distinct = new TreeSet<>();
        long id = 1L;

        for (User user : users) {
            distinct.add(user.getId());
        }

        while (true) {
            if (!distinct.contains(id)) {
                return id;
            }
            id = id + 1;
        }
    }

    @Override
    public void add(String firstName, String lastName) throws ValidationException, RepositoryException {
        Long id = getId(repository.getAll());
        User toAdd = new User(id, firstName, lastName);
        userValidator.validate(toAdd);
        repository.save(toAdd);
        network.add(toAdd);
    }

    @Override
    public User remove(long id) throws RepositoryException {
        User toDelete = repository.findAfterId(id);
        network.remove(toDelete);
        return repository.delete(id);
    }

    @Override
    public void makeFriends(long id1, long id2) throws NetworkException, ValidationException {
        Friendship friendship = new Friendship(id1, id2);
        friendshipValidator.validate(friendship);
        network.makeFriends(friendship);
    }

    @Override
    public void removeFriends(long id1, long id2) throws NetworkException, ValidationException {
        Friendship friendship = new Friendship(id1, id2);
        friendshipValidator.validate(friendship);
        network.removeFriends(friendship);
    }

    @Override
    public int numberOfCommunities() {
        return network.getNumberOfCommunities();
    }

    @Override
    public Vector<User> mostPopulatedCommunity() {
        Vector<Long> communityIds = network.getMostPopulatedCommunity();
        Vector<User> community = new Vector<>();
        for (long id : communityIds) {
            try {
                community.add(repository.findAfterId(id));
            } catch (RepositoryException e) {
                System.out.println(e.getMessage());
            }
        }
        return community;
    }

    @Override
    public Iterable<User> getAll() {
        return repository.getAll();
    }
}
