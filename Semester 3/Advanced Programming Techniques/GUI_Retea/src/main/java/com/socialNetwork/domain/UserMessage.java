package com.socialNetwork.domain;

public class UserMessage extends Message {

    private final Long currentUserId;

    public UserMessage(String text, Long fromUserId, Long toUserId, Long friendshipId, Long currentUserId) {
        super(text, fromUserId, toUserId, friendshipId);
        this.currentUserId = currentUserId;
    }

    public UserMessage(Message message, Long currentUserId) {
        super(message.getText(), message.getFromUserId(), message.getToUserId(), message.getFriendshipId());
        this.currentUserId = currentUserId;
    }

    public Long getCurrentUserId() {
        return currentUserId;
    }
}
