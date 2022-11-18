package repository.fileSystem;

import domain.Entity;
import exceptions.RepositoryException;
import repository.InMemoryRepository;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;

public abstract class AbstractFileRepository<ID, T extends Entity<ID>> extends InMemoryRepository<ID, T> {
    private final String fileName;

    public AbstractFileRepository(String fileName) {
        super();
        this.fileName = fileName;
        loadData();
    }

    private void loadData() {
        Path path = Paths.get(fileName);
        List<String> lines;
        try {
            lines = Files.readAllLines(path);
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        lines.forEach(line -> {
            T entity = extractEntity(line);
            try {
                super.save(entity);
            } catch (RepositoryException e) {
                e.printStackTrace();
            }
        });
    }

    protected abstract T extractEntity(String line);

    protected abstract String createStringFromEntity(T entity);

    private void appendToFile(T entity) throws RepositoryException {
        Path path = Paths.get(fileName);
        try (BufferedWriter bufferedWriter = Files.newBufferedWriter(path, StandardOpenOption.APPEND)) {
            String toWrite = createStringFromEntity(entity);
            bufferedWriter.write(toWrite);
            bufferedWriter.newLine();
        } catch (IOException e) {
            throw new RepositoryException(e.getMessage());
        }
    }

    private void rewriteFile() throws RepositoryException {
        Path path = Paths.get(fileName);
        Iterable<T> entities = super.getAll();

        try (BufferedWriter bufferedWriter = Files.newBufferedWriter(path, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            for (T entity : entities) {
                String toWrite = createStringFromEntity(entity);
                bufferedWriter.write(toWrite);
                bufferedWriter.newLine();
            }
        } catch (IOException e) {
            throw new RepositoryException(e.getMessage());
        }
    }

    @Override
    public void save(T entity) throws IllegalArgumentException, RepositoryException {
        super.save(entity);
        appendToFile(entity);
    }

    @Override
    public void update(ID id, T entity) throws IllegalArgumentException, RepositoryException {
        super.update(id, entity);
        rewriteFile();
    }

    @Override
    public T delete(ID id) throws RepositoryException {
        T deleted = super.delete(id);
        rewriteFile();
        return deleted;
    }
}
