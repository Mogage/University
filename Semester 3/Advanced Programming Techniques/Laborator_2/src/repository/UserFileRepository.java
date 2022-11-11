package repository;

import domain.User;
import exceptions.RepositoryException;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;

public class UserFileRepository extends InMemoryRepository<Long, User> {
    private final String fileName;

    public UserFileRepository(String fileName) {
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
            String[] words = line.split(",");
            User user = new User(words[1], words[2]);
            user.setId(Long.parseLong(words[0]));
            try {
                super.save(user);
            } catch (RepositoryException e) {
                e.printStackTrace();
            }
        });
    }

    @Override
    public void save(User entity) throws IllegalArgumentException, RepositoryException {
        super.save(entity);
        Path path = Paths.get(fileName);
        try (BufferedWriter bufferedWriter = Files.newBufferedWriter(path, StandardOpenOption.APPEND)) {
            String toWrite = entity.getId().toString() + ',' + entity.getFirstName() + ',' + entity.getLastName();
            bufferedWriter.write(toWrite);
            bufferedWriter.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void update(Long aLong, User entity) throws IllegalArgumentException, RepositoryException {
        super.update(aLong, entity);
        Path path = Paths.get(fileName);
        Iterable<User> users = super.getAll();

        try (BufferedWriter bufferedWriter = Files.newBufferedWriter(path, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            for (User user : users) {
                String toWrite = user.getId().toString() + ',' + user.getFirstName() + ',' + user.getLastName();
                bufferedWriter.write(toWrite);
                bufferedWriter.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public User delete(Long aLong) throws RepositoryException {
        User deleted = super.delete(aLong);
        Path path = Paths.get(fileName);
        Iterable<User> users = super.getAll();

        try (BufferedWriter bufferedWriter = Files.newBufferedWriter(path, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            for (User user : users) {
                String toWrite = user.getId().toString() + ',' + user.getFirstName() + ',' + user.getLastName();
                bufferedWriter.write(toWrite);
                bufferedWriter.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
            super.save(deleted);
            return null;
        }

        return deleted;
    }
}
