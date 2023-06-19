package model;

import javax.persistence.*;
import java.io.Serializable;

@javax.persistence.Entity
@Table(name = "games")
public class Game implements Entity<Integer>, Serializable {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private int id;

    @Column(name = "player")
    private String player;

    @Column(name = "date")
    private String date;

    @Column(name = "time")
    private String time;

    @Column(name = "score")
    private int score;

    public Game() {
        this.id = 0;
        this.player = "";
        this.date = "";
        this.time = "";
        this.score = 0;
    }

    public Game(String player, String date, String time, int score) {
        this.id = 0;
        this.player = player;
        this.date = date;
        this.time = time;
        this.score = score;
    }

    public Game(int id, String player, String date, String time, int score) {
        this.id = id;
        this.player = player;
        this.date = date;
        this.time = time;
        this.score = score;
    }

    @Override
    public Integer getId() {
        return id;
    }

    @Override
    public void setId(Integer id) {
        this.id = id;
    }

    public String getPlayer() {
        return player;
    }

    public void setPlayer(String player) {
        this.player = player;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public String getTime() {
        return time;
    }

    public void setTime(String time) {
        this.time = time;
    }

    public int getScore() {
        return score;
    }

    public void setScore(int score) {
        this.score = score;
    }

    @Override
    public String toString() {
        return "Game{" +
                "id=" + id +
                ", player='" + player + '\'' +
                ", date='" + date + '\'' +
                ", time='" + time + '\'' +
                ", score='" + score + '\'' +
                '}';
    }
}
