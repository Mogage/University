package clinica.repository;

import clinica.domain.Entity;

public interface Repository<ID, T extends Entity<ID>> {
    Iterable<T> getAll();

    T findAfterId(ID id);
}
