package com.socialNetwork.domain;

public class Message extends Entity<Long> {
    private String text;
    private Long fromUserId;
    private Long toUserId;

    public String getText() {
        return text;
    }

    public Long getFromUserId() {
        return fromUserId;
    }

    public Long getToUserId() {
        return toUserId;
    }

    public Message(String text, Long fromUserId, Long toUserId) {
        this.text = text;
        this.fromUserId = fromUserId;
        this.toUserId = toUserId;
    }

    @Override
    public String toString() {
        return "Message{" +
                "text='" + text + '\'' +
                ", fromUserId=" + fromUserId +
                ", toUserId=" + toUserId +
                '}';
    }
}
