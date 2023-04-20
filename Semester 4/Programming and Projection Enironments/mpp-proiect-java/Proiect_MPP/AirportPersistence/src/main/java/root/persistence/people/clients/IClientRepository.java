package root.persistence.people.clients;

import root.model.people.Client;
import root.persistence.people.IPersonRepository;

public interface IClientRepository extends IPersonRepository<Client> {
    Client findByAddress(String address);
}
