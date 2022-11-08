package userInterface;


import domain.User;
import exceptions.NetworkException;
import exceptions.RepositoryException;
import exceptions.ValidationException;
import service.Service;

import java.util.Scanner;
import java.util.Vector;

public class UI {
    private final Service<User> service;
    private final Scanner in = new Scanner(System.in);

    public UI(Service<User> service) {
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

    private void removeUser() throws RepositoryException {
        long id;

        this.printAll();
        System.out.print("Id of the user to delete: ");
        try {
            id = Long.parseLong(in.nextLine());
        } catch (NumberFormatException e) {
            return;
        }

        User removed = service.remove(id);

        System.out.println("User " + removed + " removed.");
    }

    private void addFriends() throws NetworkException, ValidationException {
        long id1;
        long id2;
        System.out.print("First user id: ");
        try {
            id1 = Long.parseLong(in.nextLine());
        } catch (NumberFormatException e) {
            return;
        }
        System.out.print("Second user id: ");
        try {
            id2 = Long.parseLong(in.nextLine());
        } catch (NumberFormatException e) {
            return;
        }

        service.makeFriends(id1, id2);

        System.out.println("Users are friends now.");
    }

    private void removeFriends() throws NetworkException, ValidationException {
        long id1;
        long id2;
        System.out.print("First user id: ");
        try {
            id1 = Long.parseLong(in.nextLine());
        } catch (NumberFormatException e) {
            return;
        }
        System.out.print("Second user id: ");
        try {
            id2 = Long.parseLong(in.nextLine());
        } catch (NumberFormatException e) {
            return;
        }

        service.removeFriends(id1, id2);

        System.out.println("Users are not friends anymore.");
    }

    private void printAll() {
        service.getAll().forEach(System.out::println);
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
                1. Add user.
                2. Remove user.
                3. Make friends.
                4. Remove friends.
                5. Show users.
                6. Show number of communities.
                7. Show most populated community.
                8. Show Menu.
                9. Exit.
                \s""");
    }

    public void run() {
        int input = 0;
        this.printMenu();
        while (true) {
            System.out.print(">>> ");
            try {
                input = Integer.parseInt(in.nextLine());
            } catch (NumberFormatException e) {
                continue;
            }
            switch (input) {
                case 1:
                    try {
                        this.addUser();
                    } catch (ValidationException | RepositoryException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                case 2:
                    try {
                        this.removeUser();
                    } catch (RepositoryException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                case 3:
                    try {
                        this.addFriends();
                    } catch (NetworkException | ValidationException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                case 4:
                    try {
                        this.removeFriends();
                    } catch (NetworkException | ValidationException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                case 5:
                    this.printAll();
                    break;
                case 6:
                    this.printNumberOfCommunities();
                    break;
                case 7:
                    try {
                        this.printPopulatedCommunity();
                    } catch (RepositoryException e) {
                        System.out.println(e.getMessage());
                    }
                    break;
                case 8:
                    this.printMenu();
                    break;
                case 9:
                    return;
                default:
                    break;
            }
        }
    }
}
