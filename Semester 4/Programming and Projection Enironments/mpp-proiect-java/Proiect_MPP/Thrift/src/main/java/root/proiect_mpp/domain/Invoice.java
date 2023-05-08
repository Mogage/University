package root.proiect_mpp.domain;

public class Invoice implements Entity<Integer> {
    private int id;
    private int clientId;

    // Class Constructors //

    public Invoice() {
        this.id = 0;
        this.clientId = 0;
    }

    public Invoice(int clientId) {
        this.id = 0;
        this.clientId = clientId;
    }

    public Invoice(int id, int clientId) {
        this.id = id;
        this.clientId = clientId;
    }

    // Getters & Setters //

    @Override
    public Integer getId() {
        return id;
    }

    @Override
    public void setId(Integer id) {
        this.id = id;
    }

    public int getClientId() {
        return clientId;
    }

    public void setClientId(int clientId) {
        this.clientId = clientId;
    }


    // toString & other functions //

    @Override
    public String toString() {
        return "Invoice{" +
                "id=" + id +
                ", clientId=" + clientId +
                '}';
    }
}
