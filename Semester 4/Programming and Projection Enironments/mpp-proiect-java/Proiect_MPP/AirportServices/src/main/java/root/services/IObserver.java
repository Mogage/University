package root.services;

import root.model.Flight;
import root.model.Ticket;

import java.util.Collection;
import java.util.List;

public interface IObserver {
    void ticketBought(Collection<Flight> flights) throws Exception;
}
