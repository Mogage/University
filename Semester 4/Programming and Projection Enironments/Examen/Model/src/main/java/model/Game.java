package model;

import org.hibernate.annotations.LazyCollection;

import javax.persistence.*;
import java.io.Serializable;
import java.util.List;
import java.util.Set;

@javax.persistence.Entity
@Table(name = "games")
public class Game implements Entity<Integer>, Serializable {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private int id;

    @Column(name = "player")
    private String player;

    @Column(name = "durata")
    private int durata;

    @Column(name = "score")
    private int score;

    @ElementCollection(fetch = FetchType.EAGER)
    @CollectionTable(name = "positions", joinColumns = @JoinColumn(name = "id"))
    @Column(name = "positions")
    private List<String> positions;

    @ElementCollection(fetch = FetchType.EAGER)
    @CollectionTable(name = "proposed_positions", joinColumns = @JoinColumn(name = "id"))
    @Column(name = "proposed_positions")
    private Set<String> proposedPositions;

    public Game() {
        this.id = 0;
        this.player = "";
        this.durata = 0;
        this.score = 0;
        this.positions = null;
        this.proposedPositions = null;
    }

    public Game(String player, int durata, int score, List<String> positions, Set<String> proposedPositions) {
        this.player = player;
        this.durata = durata;
        this.score = score;
        this.positions = positions;
        this.proposedPositions = proposedPositions;
    }

    public Game(int id, String player, int durata, int score, List<String> positions, Set<String> proposedPositions) {
        this.id = id;
        this.player = player;
        this.durata = durata;
        this.score = score;
        this.positions = positions;
        this.proposedPositions = proposedPositions;
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

    public int getDurata() {
        return durata;
    }

    public void setDurata(int durata) {
        this.durata = durata;
    }

    public int getScore() {
        return score;
    }

    public void setScore(int score) {
        this.score = score;
    }

    public List<String> getPositions() {
        return positions;
    }

    public void setPositions(List<String> positions) {
        this.positions = positions;
    }

    public Set<String> getProposedPositions() {
        return proposedPositions;
    }

    public void setProposedPositions(Set<String> proposedPositions) {
        this.proposedPositions = proposedPositions;
    }

    @Override
    public String toString() {
        return "Game{" +
                "id=" + id +
                ", player='" + player + '\'' +
                ", durata=" + durata +
                ", score=" + score +
                ", positions=" + positions +
                ", proposedPositions=" + proposedPositions +
                '}';
    }
}
