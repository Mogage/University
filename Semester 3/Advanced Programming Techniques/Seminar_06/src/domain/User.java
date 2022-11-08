package domain;

public class User extends Entity<Long> {

    private String firstName;
    private String lastName;

    public User(String firstName, String lastName) {
        this.lastName = lastName;
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    @Override
    public String toString() {
        return "User{" +
                "lastName='" + lastName + '\'' +
                ", FirstName='" + firstName + '\'' +
                '}';
    }
}
