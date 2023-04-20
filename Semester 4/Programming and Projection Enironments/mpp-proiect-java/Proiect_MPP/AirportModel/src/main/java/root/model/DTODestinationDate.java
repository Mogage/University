package root.model;

import java.io.Serializable;
import java.time.LocalDate;

public class DTODestinationDate implements Serializable {
    private final String destination;
    private final LocalDate date;

    public DTODestinationDate(String destination, LocalDate date) {
        this.destination = destination;
        this.date = date;
    }

    public String getDestination() {
        return destination;
    }

    public LocalDate getDate() {
        return date;
    }

    @Override
    public String toString() {
        return "DTODestinationDate{" +
                "destination='" + destination + '\'' +
                ", date=" + date +
                '}';
    }
}
