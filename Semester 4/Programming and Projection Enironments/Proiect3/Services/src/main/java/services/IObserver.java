package services;

import model.Game;

import java.io.IOException;
import java.util.Collection;

public interface IObserver {
    void gameFinished(Collection<Game> gamesList) throws IOException;
}
