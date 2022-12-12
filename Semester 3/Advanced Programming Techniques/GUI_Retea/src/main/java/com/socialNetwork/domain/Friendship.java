package com.socialNetwork.domain;

import com.socialNetwork.utils.Constants;

import java.time.LocalDateTime;

public class Friendship extends Entity<Long> {
    private final Long idUser1;
    private final Long idUser2;
    private final LocalDateTime friendsFrom;
    private final String status;

    public Friendship(Long idUser1, Long idUser2, LocalDateTime friendsFrom, String status) {
        this.idUser1 = idUser1;
        this.idUser2 = idUser2;
        this.friendsFrom = friendsFrom;
        this.status = status;
    }

    public Long getIdUser1() {
        return idUser1;
    }

    public Long getIdUser2() {
        return idUser2;
    }

    public LocalDateTime getFriendsFrom() {
        return friendsFrom;
    }

    public String getStatus() { return status; }

    @Override
    public String toString() {
        return "Friendship{" +
                "idFriendship=" + super.getId() +
                "idUser1=" + idUser1 +
                ", idUser2=" + idUser2 +
                ", friendsFrom=" + friendsFrom.format(Constants.DATE_TIME_FORMATTER) +
                ", status=" + status +
                '}';
    }
}
