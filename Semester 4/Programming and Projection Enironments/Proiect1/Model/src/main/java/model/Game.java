package model;

import org.hibernate.annotations.GeneratorType;

import javax.persistence.*;
import java.io.Serializable;
import java.util.List;

@javax.persistence.Entity
@Table(name = "games")
public class Game implements Entity<Integer>, Serializable {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private int id;

    @Column(name = "player")
    private String player;

    @Column(name = "indiciu")
    private String indiciu;

    @Column(name = "date")
    private String date;

    @Column(name = "time")
    private String time;

    @ElementCollection(fetch = FetchType.EAGER)
    @CollectionTable(name = "tries", joinColumns = @JoinColumn(name = "id"))
    @Column(name = "tries")
    private List<String> tries;

    public Game() {
        this.id = 0;
        this.player = "";
        this.indiciu = "";
        this.date = "";
        this.time = "";
        this.tries = null;
    }

    public Game(String player, String indiciu, String date, String time, List<String> tries) {
        this.id = 0;
        this.player = player;
        this.indiciu = indiciu;
        this.date = date;
        this.time = time;
        this.tries = tries;
    }

    public Game(int id, String player, String indiciu, String date, String time, List<String> tries) {
        this.id = id;
        this.player = player;
        this.indiciu = indiciu;
        this.date = date;
        this.time = time;
        this.tries = tries;
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

    public String getIndiciu() {
        return indiciu;
    }

    public void setIndiciu(String indiciu) {
        this.indiciu = indiciu;
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

    public List<String> getTries() {
        return tries;
    }

    public void setTries(List<String> tries) {
        this.tries = tries;
    }

    @Override
    public String toString() {
        return "Game{" +
                "id=" + id +
                ", player='" + player + '\'' +
                ", indiciu=" + indiciu +
                ", date='" + date + '\'' +
                ", time='" + time + '\'' +
                ", tries=" + tries +
                '}';
    }
}
