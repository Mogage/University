package com.socialNetwork.domain;

import com.socialNetwork.utils.Constants;

import java.time.LocalDateTime;

public class DTOUserFriendship {
    private Long id;
    private String firstName;
    private String lastName;
    private LocalDateTime friendsFrom;

    public Long getId() {
        return id;
    }

    public String getFirstName() {
        return firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public String getFriendsFrom() {
        return friendsFrom.format(Constants.DATE_TIME_FORMATTER);
    }

    public DTOUserFriendship(Long id, String firstName, String lastName, LocalDateTime friendsFrom) {
        this.id = id;
        this.firstName = firstName;
        this.lastName = lastName;
        this.friendsFrom = friendsFrom;
    }
}
