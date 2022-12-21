package com.socialNetwork.repository.databaseSystem;

import com.socialNetwork.domain.Message;
import com.socialNetwork.exceptions.RepositoryException;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public class MessagesDBRepository extends AbstractDBRepository<Long, Message>{
    public MessagesDBRepository(String url, String userName, String password) {
        super(url, userName, password, "SELECT * FROM messages");
    }

    @Override
    protected Message extractEntity(ResultSet resultSet) throws SQLException {
        Long messageID = resultSet.getLong("id");
        String text = resultSet.getString("text");
        Long fromUserId = resultSet.getLong("from_user");
        Long toUserId = resultSet.getLong("to_user");
        Long friendshipId = resultSet.getLong("id_friendship");
        Message message = new Message(text, fromUserId, toUserId, friendshipId);
        message.setId(messageID);
        return message;
    }

    @Override
    protected PreparedStatement createInsertStatement(Connection connection, Message entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.getSqlCommand());
        statement.setLong(1, entity.getId());
        statement.setString(2, entity.getText());
        statement.setLong(3, entity.getFromUserId());
        statement.setLong(4, entity.getToUserId());
        statement.setLong(5, entity.getFriendshipId());
        return statement;
    }

    @Override
    protected PreparedStatement createUpdateStatement(Connection connection, Message entity) throws SQLException {
        return null;
    }

    @Override
    protected PreparedStatement createDeleteStatement(Connection connection, Message entity) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(super.getSqlCommand());
        statement.setLong(1, entity.getId());
        return statement;
    }

    @Override
    public void save(Message entity) throws RepositoryException {
        setSqlCommand("INSERT INTO messages(id, text, from_user, to_user, id_friendship) VALUES (?, ?, ?, ?, ?)");
        super.save(entity);
    }

    @Override
    public Message delete(Long aLong) throws RepositoryException {
        setSqlCommand("DELETE FROM messages WHERE id=?");
        return super.delete(aLong);
    }
}
