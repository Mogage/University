package repository.fileSystem;

import domain.Friendship;

import java.time.LocalDateTime;
import java.util.Objects;
import java.util.Vector;

public class FriendshipFileRepository extends AbstractFileRepository<Long, Friendship> {
    public FriendshipFileRepository(String fileName) {
        super(fileName);
    }

    @Override
    protected Friendship extractEntity(String line) {
        String[] ids = line.split(",");
        Friendship friendship = new Friendship(Long.parseLong(ids[1]), Long.parseLong(ids[2]), LocalDateTime.parse(ids[3]));
        friendship.setId(Long.parseLong(ids[0]));
        return friendship;
    }

    @Override
    protected String createStringFromEntity(Friendship entity) {
        return entity.getId().toString() + ',' +
                entity.getIdUser1() + ',' +
                entity.getIdUser2() + ',' +
                entity.getFriendsFrom();
    }

    public Vector<Friendship> findUserFriends(Long id) {
        Iterable<Friendship> friendships = super.getAll();
        Vector<Friendship> result = new Vector<>();

        for (Friendship friendship : friendships) {
            if (Objects.equals(friendship.getIdUser1(), id) || Objects.equals(friendship.getIdUser2(), id)) {
                result.add(friendship);
            }
        }

        return result;
    }
}
