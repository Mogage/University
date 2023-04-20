package root.services;

import root.model.Ticket;

public interface IObserver {
    void ticketBought(Ticket ticket) throws Exception;
}
