package services;

import model.DtoInitialise;
import model.Game;
import model.Player;

public interface IService {
    Player login(Player player, IObserver client) throws Exception;
    DtoInitialise initialise(int id) throws Exception;
    int move(Integer id) throws Exception;
    int getScore(Integer id) throws Exception;
    void logout(Player player);
}
