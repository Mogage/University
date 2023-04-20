package root.model.people;

public class Employee extends Person{
    private String position;
    private String email;
    private String password;

    // Class Constructors //

    public Employee() {
        super();
        this.position = "";
    }

    public Employee(String firstName, String lastName, String position, String email, String password) {
        super(firstName, lastName);
        this.position = position;
        this.email = email;
        this.password = password;
    }

    public Employee(int id, String firstName, String lastName, String position, String email, String password) {
        super(id, firstName, lastName);
        this.position = position;
        this.email = email;
        this.password = password;
    }

    public Employee(String email, String password) {
        super();
        this.position = "";
        this.email = email;
        this.password = password;
    }

    // Getter & Setter //

    public String getPosition() {
        return position;
    }

    public void setPosition(String position) {
        this.position = position;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    // toString & other functions //

    @Override
    public String toString() {
        return "Employee{" + super.toString() +
                "position='" + position + '\'' +
                ", email='" + email + '\'' +
                ", password='" + password + '\'' +
                '}';
    }
}
