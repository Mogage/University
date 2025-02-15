package repository;

import domain.Entity;
import exceptions.RepositoryException;

import java.util.HashMap;
import java.util.Map;

public abstract class InMemoryRepository<ID, T extends Entity<ID>> implements Repository<ID, T> {
    private final Map<ID, T> entities;

    public InMemoryRepository() {
        entities = new HashMap<>();
    }

    @Override
    public void save(T entity) throws IllegalArgumentException, RepositoryException {
        if (entity == null) {
            throw new IllegalArgumentException("Entity cannot be null");
        }
        if (entities.containsKey(entity.getId())) {
            throw new RepositoryException("Element with this id already exists.\n");
        }
        if (entities.containsValue(entity)) {
            throw new RepositoryException("This element is already added.\n");
        }
        entities.put(entity.getId(), entity);
    }

    @Override
    public void update(ID id, T entity) throws IllegalArgumentException, RepositoryException {
        if (entity == null) {
            throw new IllegalArgumentException("Entity cannot be null");
        }
        if (!entities.containsKey(id)) {
            throw new RepositoryException("Element with this id does not exist.\n");
        }
        entities.remove(id);
        entities.put(entity.getId(), entity);
    }

    @Override
    public T delete(ID id) throws RepositoryException {
        if (!entities.containsKey(id)) {
            throw new RepositoryException("Element with this id does not exist.\n");
        }
        return entities.remove(id);
    }

    @Override
    public T findAfterId(ID id) throws RepositoryException {
        if (!entities.containsKey(id)) {
            throw new RepositoryException("Element with this id does not exist.\n");
        }
        return entities.get(id);
    }

    @Override
    public Iterable<T> getAll() {
        return entities.values();
    }

    @Override
    public int size() {
        return entities.size();
    }
}
