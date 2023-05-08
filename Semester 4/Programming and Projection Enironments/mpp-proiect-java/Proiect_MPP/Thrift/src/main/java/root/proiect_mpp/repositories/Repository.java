package root.proiect_mpp.repositories;

import java.util.Collection;

public interface Repository<T, Tid> {
    int add(T elem);
    void delete(T elem);
    void update(T elem, Tid id);
    T findById(Tid id);
    Collection<T> getAll();
}
