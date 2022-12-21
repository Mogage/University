package com.socialNetwork.domain;

public class Message extends Entity<Long> {
    private String text;
    private Long fromUserId;
    private Long toUserId;
    private Long friendshipId;

    public String getText() {
        return text;
    }

    public Long getFromUserId() {
        return fromUserId;
    }

    public Long getToUserId() {
        return toUserId;
    }

    public Long getFriendshipId() {
        return friendshipId;
    }

    public Message(String text, Long fromUserId, Long toUserId, Long friendshipId) {
        this.text = text;
        this.fromUserId = fromUserId;
        this.toUserId = toUserId;
        this.friendshipId = friendshipId;
    }

    @Override
    public String toString() {
        return text;
    }
}
