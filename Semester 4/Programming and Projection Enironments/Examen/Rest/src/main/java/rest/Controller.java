package rest;

import model.Game;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import persistence.games.GameRepository;

import java.util.Collection;

@CrossOrigin
@RestController
public class Controller {
    @Autowired
    private GameRepository gameRepository;

    @RequestMapping("/greeting")
    public String greeting() {
        return "Hello, World!";
    }

    @RequestMapping(method = RequestMethod.GET)
    public Collection<Game> getAll() {
        return gameRepository.getAll();
    }

    @RequestMapping(value = "/games/{username}", method = RequestMethod.GET)
    public Collection<Game> getByUsername(@PathVariable String username) {
        return gameRepository.getByUsername(username);
    }

    @RequestMapping(value = "/{idGame}", method = RequestMethod.POST)
    public Game create(@PathVariable String idGame, @RequestBody String position) {
        Game game = gameRepository.findById(Integer.parseInt(idGame));
        if (game == null) {
            return null;
        }
        game.getProposedPositions().add(position);
        gameRepository.updateGame(game);
        return game;
    }
}
