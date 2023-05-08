package root.model;

import java.io.Serializable;

public class Ticket implements Entity<Integer>, Serializable {
    private int id;
    private int flightId;
    private int invoiceId;
    private int seatNumber;
    private String touristName;

    // Class constructors //

    public Ticket() {
        this.id = 0;
        this.flightId = 0;
        this.invoiceId = 0;
        this.seatNumber = 0;
        this.touristName = "";
    }

    public Ticket(int flightId, int invoiceId, int seatNumber, String touristName) {
        this.id = 0;
        this.flightId = flightId;
        this.invoiceId = invoiceId;
        this.seatNumber = seatNumber;
        this.touristName = touristName;
    }

    public Ticket(int id, int flightId, int invoiceId, int seatNumber, String touristName) {
        this.id = id;
        this.flightId = flightId;
        this.invoiceId = invoiceId;
        this.seatNumber = seatNumber;
        this.touristName = touristName;
    }

    // Getters & Setters //

    @Override
    public Integer getId() {
        return this.id;
    }

    @Override
    public void setId(Integer id) {
        this.id = id;
    }

    public int getFlightId() {
        return flightId;
    }

    public void setFlightId(int flightId) {
        this.flightId = flightId;
    }

    public int getInvoiceId() {
        return invoiceId;
    }

    public void setInvoiceId(int invoiceId) {
        this.invoiceId = invoiceId;
    }

    public int getSeatNumber() {
        return seatNumber;
    }

    public void setSeatNumber(int seatNumber) {
        this.seatNumber = seatNumber;
    }

    public String getTouristName() {
        return touristName;
    }

    public void setTouristName(String touristName) {
        this.touristName = touristName;
    }

    // toString & other functions //

    @Override
    public String toString() {
        return "Ticket{" +
                "id=" + id +
                ", flightId=" + flightId +
                ", invoiceId=" + invoiceId +
                ", seatNumber=" + seatNumber +
                ", touristId=" + touristName +
                '}';
    }
}
