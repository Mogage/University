package com.socialNetwork.repository.databaseSystem;

import com.socialNetwork.domain.Friendship;
import com.socialNetwork.exceptions.RepositoryException;

import java.sql.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Objects;
import java.util.List;

public class FriendshipDBRepository extends AbstractDBRepository<Long, Friendship> {
    public FriendshipDBRepository(String url, String userName, String password) {
        super(url, userName, password, "SELECT * FROM friendships");
    }

    @Override
    protected Friendship extractEntity(ResultSet resultSet) throws SQLException {
        Long idFriendship = resultSet.getLong("id_friendship");
        Long idUser1 = resultSet.getLong("id_user1");
        Long idUser2 = resultSet.getLong("id_user2");
        String status = resultSet.getString("status");
        LocalDateTime friendsFrom = LocalDateTime.parse(resultSet.getString("friends_from"));
        Friendship friendship = new Friendship(idUser1, idUser2, friendsFrom, status);
        friendship.setId(idFriendship);
        return friendship;
    }

    @Override
    protected PreparedStatement createInsertStatement(Connection connection, Friendship entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.getSqlCommand());
        statement.setLong(1, entity.getId());
        statement.setLong(2, entity.getIdUser1());
        statement.setLong(3, entity.getIdUser2());
        statement.setString(4, entity.getFriendsFrom().toString());
        statement.setString(5, entity.getStatus());
        return statement;
    }

    @Override
    protected PreparedStatement createUpdateStatement(Connection connection, Friendship entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.getSqlCommand());
        statement.setLong(1, entity.getIdUser1());
        statement.setLong(2, entity.getIdUser2());
        statement.setString(3, entity.getFriendsFrom().toString());
        statement.setString(4, entity.getStatus());
        statement.setLong(5, entity.getId());
        return statement;
    }

    @Override
    protected PreparedStatement createDeleteStatement(Connection connection, Friendship entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.getSqlCommand());
        statement.setLong(1, entity.getId());
        return statement;
    }

    @Override
    public void save(Friendship entity) throws IllegalArgumentException, RepositoryException {
        super.setSqlCommand("INSERT INTO Friendships (id_friendship, id_user1, id_user2, friends_from, status) VALUES (?, ?, ?, ?, ?)");
        super.save(entity);
    }

    @Override
    public void update(Long aLong, Friendship entity) throws IllegalArgumentException, RepositoryException {
        super.setSqlCommand("UPDATE Friendships SET id_user1=?, id_user2=?, friends_from=?, status=? WHERE id_friendship=?");
        super.update(aLong, entity);
    }

    @Override
    public Friendship delete(Long aLong) throws RepositoryException {
        super.setSqlCommand("DELETE FROM Friendships WHERE id_friendship=?");
        return super.delete(aLong);
    }

    public List<Friendship> findUserFriends(Long id) {
        Iterable<Friendship> friendships = super.getAll();
        List<Friendship> result = new ArrayList<>();

        for (Friendship friendship : friendships) {
            if (Objects.equals(friendship.getIdUser1(), id) || Objects.equals(friendship.getIdUser2(), id)) {
                result.add(friendship);
            }
        }

        return result;
    }
}