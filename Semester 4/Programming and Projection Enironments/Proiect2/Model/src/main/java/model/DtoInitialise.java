package model;

import java.io.Serializable;
import java.util.Collection;

public class DtoInitialise implements Serializable {
    private Configuration configuration;
    private Collection<Game> scores;

    public DtoInitialise() {
        this.configuration = null;
        this.scores = null;
    }

    public DtoInitialise(Configuration configuration, Collection<Game> scores) {
        this.configuration = configuration;
        this.scores = scores;
    }

    public Configuration getConfiguration() {
        return configuration;
    }

    public void setConfiguration(Configuration configuration) {
        this.configuration = configuration;
    }

    public Collection<Game> getScores() {
        return scores;
    }

    public void setScores(Collection<Game> scores) {
        this.scores = scores;
    }

    @Override
    public String toString() {
        return "DtoInitialise{" + "configuration=" + configuration + ", scores=" + scores + '}';
    }
}
