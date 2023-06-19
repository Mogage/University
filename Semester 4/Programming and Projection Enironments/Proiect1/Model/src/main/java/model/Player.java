package model;

import java.io.Serializable;

public class Player implements Entity<Integer>, Serializable {
    private int id;
    private String username;

    public Player() {
        this.id = 0;
        this.username = "";
    }

    public Player(String username) {
        this.id = 0;
        this.username = username;
    }

    public Player(int id, String username) {
        this.id = id;
        this.username = username;
    }

    @Override
    public Integer getId() {
        return id;
    }

    @Override
    public void setId(Integer id) {
        this.id = id;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
    	this.username = username;
    }

    @Override
    public String toString() {
        return "Player{" +
                "id=" + id +
                ", username='" + username + '\'' +
                '}';
    }
}
