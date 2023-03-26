package root.proiect_mpp.repositories.people.clients;

import root.proiect_mpp.domain.people.Client;
import root.proiect_mpp.repositories.people.IPersonRepository;

public interface IClientRepository extends IPersonRepository<Client> {
    Client findByAddress(String address);
}
