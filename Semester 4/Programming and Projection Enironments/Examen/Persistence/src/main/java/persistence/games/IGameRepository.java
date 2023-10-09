package persistence.games;

import model.Game;

import java.util.Collection;
import java.util.List;

public interface IGameRepository {
    Game startGame(String player, List<String> positions);
    void updateGame(Game game);
    Collection<Game> getAll();
    Collection<Game> getByUsername(String username);
    Game findById(int parseInt);
}
