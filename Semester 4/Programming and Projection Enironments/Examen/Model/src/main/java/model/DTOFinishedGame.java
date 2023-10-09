package model;

import java.io.Serializable;
import java.util.List;

public class DTOFinishedGame implements Serializable {
    private int score;
    private List<String> holesPosition;

    public DTOFinishedGame() {
        this.score = 0;
        this.holesPosition = null;
    }

    public DTOFinishedGame(int score, List<String> holesPosition) {
        this.score = score;
        this.holesPosition = holesPosition;
    }

    public int getScore() {
        return score;
    }

    public void setScore(int score) {
        this.score = score;
    }

    public List<String> getHolesPosition() {
        return holesPosition;
    }

    public void setHolesPosition(List<String> holesPosition) {
        this.holesPosition = holesPosition;
    }

    @Override
    public String toString() {
        return "DTOFinishedGame{" + "score=" + score + ", holesPosition=" + holesPosition + '}';
    }
}
