package domain;

import utils.Constants;

import java.time.LocalDateTime;

public class Friendship extends Entity<Long> {
    private final long idUser1;
    private final long idUser2;
    private final LocalDateTime friendsFrom;

    public Friendship(long idUser1, long idUser2, LocalDateTime friendsFrom) {
        this.idUser1 = idUser1;
        this.idUser2 = idUser2;
        this.friendsFrom = friendsFrom;
    }

    public long getIdUser1() {
        return idUser1;
    }

    public long getIdUser2() {
        return idUser2;
    }

    public LocalDateTime getFriendsFrom() {
        return friendsFrom;
    }

    @Override
    public String toString() {
        return "Friendship{" +
                "idUser1=" + idUser1 +
                ", idUser2=" + idUser2 +
                ", friendsFrom=" + friendsFrom.format(Constants.DATE_TIME_FORMATTER) +
                '}';
    }
}
