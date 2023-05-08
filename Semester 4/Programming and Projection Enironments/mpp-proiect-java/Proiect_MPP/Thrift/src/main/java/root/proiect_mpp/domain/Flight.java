package root.proiect_mpp.domain;

import java.time.LocalDate;
import java.time.LocalTime;

public class Flight implements Entity<Integer> {
    private int id;
    private int freeSeats;
    private int destinationAirport;
    private int departureAirport;
    private LocalDate departureDate;
    private LocalTime departureTime;

    // Class Constructors //

    public Flight() {
        this.id = 0;
        this.freeSeats = 0;
        this.destinationAirport = 0;
        this.departureAirport = 0;
        this.departureDate = LocalDate.now();
        this.departureTime = LocalTime.now();
    }

    public Flight(int freeSeats, int destinationAirport, int departureAirport, LocalDate departureDate, LocalTime departureTime) {
        this.id = 0;
        this.freeSeats = freeSeats;
        this.destinationAirport = destinationAirport;
        this.departureAirport = departureAirport;
        this.departureDate = departureDate;
        this.departureTime = departureTime;
    }

    public Flight(int id, int freeSeats, int destinationAirport, int departureAirport, LocalDate departureDate, LocalTime departureTime) {
        this.id = id;
        this.freeSeats = freeSeats;
        this.destinationAirport = destinationAirport;
        this.departureAirport = departureAirport;
        this.departureDate = departureDate;
        this.departureTime = departureTime;
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

    public int getFreeSeats() {
        return freeSeats;
    }

    public void setFreeSeats(int freeSeats) {
        this.freeSeats = freeSeats;
    }

    public int getDestinationAirport() {
        return destinationAirport;
    }

    public void setDestinationAirport(int destinationAirport) {
        this.destinationAirport = destinationAirport;
    }

    public int getDepartureAirport() {
        return departureAirport;
    }

    public void setDepartureAirport(int departureAirport) {
        this.departureAirport = departureAirport;
    }

    public LocalDate getDepartureDate() {
        return departureDate;
    }

    public void setDepartureDate(LocalDate departureDate) {
        this.departureDate = departureDate;
    }

    public LocalTime getDepartureTime() {
        return departureTime;
    }

    public void setDepartureTime(LocalTime departureTime) {
        this.departureTime = departureTime;
    }

    // toString & other functions //

    @Override
    public String toString() {
        return "Flight{" +
                "id=" + id +
                ", freeSeats=" + freeSeats +
                ", destination airport id='" + destinationAirport + '\'' +
                ", departure airport id='" + departureAirport + '\'' +
                ", departureDate=" + departureDate +
                ", departureTime=" + departureTime +
                '}';
    }
}
