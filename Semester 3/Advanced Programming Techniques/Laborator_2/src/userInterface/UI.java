package userInterface;

import domain.Friendship;
import domain.User;
import exceptions.NetworkException;
import exceptions.RepositoryException;
import exceptions.ValidationException;
import service.Service;
import utils.Constants;

import java.util.Scanner;
import java.util.Vector;

public class UI {
    private final Service service;
    private final Scanner in = new Scanner(System.in);

    public UI(Service service) {
        this.service = service;
    }

    private void addUser() throws ValidationException, RepositoryException {
        String firstName;
        String lastName;
        System.out.print("First name: ");
        firstName = in.nextLine();
        System.out.print("Last name: ");
        lastName = in.nextLine();

        service.add(firstName, lastName);

        System.out.println("User added.");
    }

    private void updateUser() throws ValidationException, RepositoryException {
        long id;
        String firstName;
        String lastName;

        if (0 == service.numberOfUsers()) {
            System.out.println("There are no users.");
            return;
        }

        this.printAllUsers();
        System.out.print("Id of the user to update: ");
        try {
            id = Long.parseLong(in.nextLine());
        } catch (NumberFormatException e) {
            return;
        }
        System.out.print("New first name: ");
        firstName = in.nextLine();
        System.out.print("New last name: ");
        lastName = in.nextLine();

        service.updateUser(id, firstName, lastName);

        System.out.println("User updated.");

    }

    private void removeUser() throws RepositoryException, NetworkException {
        long id;

        if (0 == service.numberOfUsers()) {
            System.out.println("There are no users.");
            return;
        }

        this.printAllUsers();
        System.out.print("Id of the user to delete: ");
        try {
            id = Long.parseLong(in.nextLine());
        } catch (NumberFormatException e) {
            return;
        }

        User removed = service.remove(id);

        System.out.println("User " + removed + " removed.");
    }

    private Long[] readIds() {
        Long[] ids = new Long[2];
        System.out.print("First user id: ");
        try {
            ids[0] = Long.parseLong(in.nextLine());
        } catch (NumberFormatException e) {
            return null;
        }
        System.out.print("Second user id: ");
        try {
            ids[1] = Long.parseLong(in.nextLine());
        } catch (NumberFormatException e) {
            return null;
        }

        return ids;
    }

    private void addFriends() throws NetworkException, ValidationException, RepositoryException {
        if (2 > service.numberOfUsers()) {
            System.out.println("A friendship cannot be made at the moment.");
            return;
        }

        Long[] ids = readIds();

        if (null == ids) {
            return;
        }
        service.makeFriends(ids[0], ids[1]);

        System.out.println("Users are friends now.");
    }

    private void updateFriends() throws ValidationException, RepositoryException, NetworkException {
        long id;

        if (0 == service.numberOfFriendships()) {
            System.out.println("There are no friends.");
            return;
        }

        this.printAllFriendships();
        System.out.print("Id of the friendship to update: ");
        try {
            id = Long.parseLong(in.nextLine());
        } catch (NumberFormatException e) {
            return;
        }
        Long[] ids = readIds();

        if (null == ids) {
            return;
        }
        service.updateFriends(id, ids[0], ids[1]);

        System.out.println("Friendship updated.");
    }

    private void removeFriends() throws NetworkException, ValidationException, RepositoryException {
        long id;

        if (0 == service.numberOfFriendships()) {
            System.out.println("There are no friends.");
            return;
        }

        this.printAllFriendships();
        System.out.print("Id of the friendship to delete: ");
        try {
            id = Long.parseLong(in.nextLine());
        } catch (NumberFormatException e) {
            return;
        }

        service.removeFriends(id);

        System.out.println("Users are not friends anymore.");
    }

    private void printAllUsers() {
        service.getAllUsers().forEach(System.out::println);
    }

    private void printAllFriendships() throws RepositoryException {
        Iterable<Friendship> friendships = service.getAllFriendships();
        for (Friendship friendship : friendships) {
            User user1 = service.getUser(friendship.getIdUser1());
            User user2 = service.getUser(friendship.getIdUser2());
            String toPrint = "Id: " + friendship.getId() + " | Friends: " +
                    user1.getFirstName() + " " +
                    user1.getLastName() + " - " +
                    user2.getFirstName() + " " +
                    user2.getLastName() + " since: " +
                    friendship.getFriendsFrom().format(Constants.DATE_TIME_FORMATTER);
            System.out.println(toPrint);
        }
    }

    private void printNumberOfCommunities() {
        System.out.println("Number of communities: " + service.numberOfCommunities());
    }

    private void printPopulatedCommunity() throws RepositoryException {
        Vector<User> community = service.mostPopulatedCommunity();
        for (User user : community) {
            System.out.println(user);
        }
    }

    private void printMenu() {
        System.out.print("""
                Menu app:
                0. Exit.
                1. Show Menu.
                2. Add user.
                3. Update user.
                4. Remove user.
                5. Make friends.
                6. Update friends.
                7. Remove friends.
                8. Show users.
                9. Show friendships.
                10. Show number of communities.
                11. Show most populated community.
                \s""");
    }

    public void runMain() {
        int input;
        this.printMenu();
        while (true) {
            System.out.print(">>> ");
            try {
                input = Integer.parseInt(in.nextLine());
            } catch (NumberFormatException e) {
                continue;
            }
            switch (input) {
                case 0:
                    return;
                case 1:
                    this.printMenu();
                    break;
                case 2:
                    try {
                        this.addUser();
                    } catch (ValidationException | RepositoryException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                case 3:
                    try {
                        this.updateUser();
                    } catch (ValidationException | RepositoryException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                case 4:
                    try {
                        this.removeUser();
                    } catch (RepositoryException | NetworkException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                case 5:
                    try {
                        this.addFriends();
                    } catch (NetworkException | ValidationException | RepositoryException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                case 6:
                    try {
                        this.updateFriends();
                    } catch (ValidationException | RepositoryException | NetworkException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                case 7:
                    try {
                        this.removeFriends();
                    } catch (NetworkException | ValidationException | RepositoryException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                case 8:
                    this.printAllUsers();
                    break;
                case 9:
                    try {
                        this.printAllFriendships();
                    } catch (RepositoryException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                case 10:
                    this.printNumberOfCommunities();
                    break;
                case 11:
                    try {
                        this.printPopulatedCommunity();
                    } catch (RepositoryException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                default:
                    break;
            }
        }
    }
}
