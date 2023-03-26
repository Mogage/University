package root.proiect_mpp.domain.people;

public class Client extends Person {
    private String address;

    // Class constructors //

    public Client () {
        super();
        address = "";
    }

    public Client(String firstName, String lastName, String address) {
        super(firstName, lastName);
        this.address = address;
    }

    public Client(int id, String firstName, String lastName, String address) {
        super(id, firstName, lastName);
        this.address = address;
    }

    // Getter & Setter //

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    // toString & other functions //

    @Override
    public String toString() {
        return "Client{" + super.toString() +
                "address='" + address + '\'' +
                '}';
    }
}
