package root.proiect_mpp.domain;

import root.proiect_mpp.utils.Constants;

import java.time.LocalDate;
import java.time.LocalTime;

public class DTOAirportFlight implements Entity<Integer> {
    private int id;
    private String departureName;
    private String departureCityName;
    private String destinationName;
    private String destinationCityName;
    private LocalDate departureDate;
    private LocalTime departureTime;
    private int freeSeats;

    public DTOAirportFlight(int id, String departureCityName, String departureName, String destinationCityName,
                            String destinationName, LocalDate departureDate, LocalTime departureTime, int freeSeats) {
        this.id = id;
        this.departureName = departureName;
        this.departureCityName = departureCityName;
        this.destinationName = destinationName;
        this.destinationCityName = destinationCityName;
        this.departureDate = departureDate;
        this.departureTime = departureTime;
        this.freeSeats = freeSeats;
    }

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

    public String getDepartureName() {
        return departureName;
    }

    public void setDepartureName(String departureName) {
        this.departureName = departureName;
    }

    public String getDepartureCityName() {
        return departureCityName;
    }

    public void setDepartureCityName(String departureCityName) {
        this.departureCityName = departureCityName;
    }

    public String getDestinationName() {
        return destinationName;
    }

    public void setDestinationName(String destinationName) {
        this.destinationName = destinationName;
    }

    public String getDestinationCityName() {
        return destinationCityName;
    }

    public void setDestinationCityName(String destinationCityName) {
        this.destinationCityName = destinationCityName;
    }

    public String getDepartureDate() {
        return departureDate.format(Constants.DATE_FORMATTER);
    }

    public void setDepartureDate(LocalDate departureDate) {
        this.departureDate = departureDate;
    }

    public String getDepartureTime() {
        return departureTime.format(Constants.TIME_FORMATTER);
    }

    public void setDepartureTime(LocalTime departureTime) {
        this.departureTime = departureTime;
    }
}
