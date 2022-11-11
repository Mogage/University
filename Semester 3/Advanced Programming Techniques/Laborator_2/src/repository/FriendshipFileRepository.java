package repository;

import domain.Friendship;
import exceptions.RepositoryException;
import utils.Constants;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Vector;

public class FriendshipFileRepository extends InMemoryRepository<Long, Friendship> {
    private final String fileName;

    public FriendshipFileRepository(String fileName) {
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
            String[] ids = line.split(",");
            Friendship friendship = new Friendship(Long.parseLong(ids[1]), Long.parseLong(ids[2]), LocalDateTime.now());
            friendship.setId(Long.parseLong(ids[0]));
            try {
                super.save(friendship);
            } catch (RepositoryException e) {
                e.printStackTrace();
            }
        });
    }

    public Vector<Friendship> findIdFriendships(Long id) {
        Iterable<Friendship> friendships = super.getAll();
        Vector<Friendship> result = new Vector<>();

        for (Friendship friendship : friendships) {
            if (friendship.getIdUser1() == id || friendship.getIdUser2() == id) {
                result.add(friendship);
            }
        }

        return result;
    }

    @Override
    public void save(Friendship entity) throws IllegalArgumentException, RepositoryException {
        super.save(entity);
        Path path = Paths.get(fileName);
        try (BufferedWriter bufferedWriter = Files.newBufferedWriter(path, StandardOpenOption.APPEND)) {
            String toWrite = entity.getId().toString() + ',' +
                    entity.getIdUser1() + ',' +
                    entity.getIdUser2() + ',' +
                    entity.getFriendsFrom().format(Constants.DATE_TIME_FORMATTER);
            bufferedWriter.write(toWrite);
            bufferedWriter.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void writeFriendships() throws IOException {
        Path path = Paths.get(fileName);
        Iterable<Friendship> friendships = super.getAll();

        BufferedWriter bufferedWriter = Files.newBufferedWriter(path, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
        for (Friendship friendship : friendships) {
            String toWrite = friendship.getId().toString() + ',' +
                    friendship.getIdUser1() + ',' +
                    friendship.getIdUser2() + ',' +
                    friendship.getFriendsFrom().format(Constants.DATE_TIME_FORMATTER);
            bufferedWriter.write(toWrite);
            bufferedWriter.newLine();
        }
    }

    @Override
    public void update(Long aLong, Friendship entity) throws IllegalArgumentException, RepositoryException {
        super.update(aLong, entity);

        try {
            writeFriendships();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public Friendship delete(Long aLong) throws RepositoryException {
        Friendship deleted = super.delete(aLong);

        try {
            writeFriendships();
        } catch (IOException e) {
            e.printStackTrace();
            super.save(deleted);
            return null;
        }

        return deleted;
    }
}
