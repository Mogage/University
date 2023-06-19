package server.implementations;

import model.Configuration;
import model.DtoInitialise;
import model.Game;
import model.Player;
import persistence.configurations.IConfigurationRepository;
import persistence.games.IGameRepository;
import persistence.players.IPlayerRepository;
import services.IObserver;
import services.IService;

import java.time.LocalDate;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class ServiceImplementation implements IService {
    IPlayerRepository playerRepository;
    IConfigurationRepository configurationRepository;
    IGameRepository gameRepository;
    private final Map<Integer, IObserver> loggedPlayers;
    private final Map<Integer, Game> games;
    private final Map<Integer, Position> positions;
    private final Map<Integer, Configuration> configurations;


    public ServiceImplementation(IPlayerRepository playerRepository, IConfigurationRepository configurationRepository,
                                 IGameRepository gameRepository) {
        this.playerRepository = playerRepository;
        this.configurationRepository = configurationRepository;
        this.gameRepository = gameRepository;
        loggedPlayers = new ConcurrentHashMap<>();
        games = new HashMap<>();
        positions = new HashMap<>();
        configurations = new HashMap<>();
    }

    @Override
    public Player login(Player player, IObserver client) throws Exception {
        Player playerToLogin = playerRepository.findByUsername(player.getUsername());
        if (playerToLogin == null) {
            throw new Exception("Authentication failed.");
        }
        if (loggedPlayers.get(playerToLogin.getId()) != null) {
            throw new Exception("Player already logged in.");
        }
        loggedPlayers.put(playerToLogin.getId(), client);

        Configuration configuration = configurationRepository.getRandomConfiguration();

        Game game = gameRepository.startGame(playerToLogin.getUsername(),
                LocalDate.now().format(DateTimeFormatter.ofPattern("dd/MM/yyyy")),
                LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss")));

        games.put(playerToLogin.getId(), game);
        positions.put(playerToLogin.getId(), new Position());
        configurations.put(playerToLogin.getId(), configuration);

        return playerToLogin;
    }

    @Override
    public void logout(Player player) {
        loggedPlayers.remove(player.getId());
    }

    @Override
    public DtoInitialise initialise(int id) {
        Collection<Game> gamesList = gameRepository.getAll();
        for (Game game1 : games.values()) {
            gamesList.removeIf(game2 -> Objects.equals(game1.getId(), game2.getId()));
        }
        return new DtoInitialise(configurations.get(id), gamesList);
    }

    @Override
    public int move(Integer id) {
        Random random = new Random();
        Position position = positions.get(id);

        if (position.getTries() == 3) {
            return 0;
        }

        int x = random.nextInt(3) + 1;
        int nextPosition = (position.getCurrentPosition() + x) % 5;
        position.setTries(position.getTries() + 1);
        Game game = games.get(id);
        if (nextPosition < position.getCurrentPosition()) {
            game.setScore(game.getScore() + 5);
        }
        if (!position.getVisited().get(nextPosition)) {
            Configuration configuration = configurations.get(id);
            int value = switch (nextPosition) {
                case 0 -> configuration.getValue1();
                case 1 -> configuration.getValue2();
                case 2 -> configuration.getValue3();
                case 3 -> configuration.getValue4();
                default -> configuration.getValue5();
            };
            game.setScore(game.getScore() - value);
        }
        gameRepository.updateGame(game);
        games.put(id, game);
        position.setCurrentPosition(nextPosition);
        position.getVisited().set(nextPosition, true);
        positions.put(id, position);
        return nextPosition + 1;
    }

    @Override
    public int getScore(Integer id) {
        int output = games.get(id).getScore();

        games.remove(id);
        configurations.remove(id);
        positions.remove(id);

        Collection<Game> gamesList = gameRepository.getAll();
        for (Game game1 : games.values()) {
            gamesList.removeIf(game2 -> Objects.equals(game1.getId(), game2.getId()));
        }
        for (IObserver loggedPlayer : loggedPlayers.values()) {
            try {
                loggedPlayer.gameFinished(gamesList);
            } catch (Exception e) {
                System.out.println("Error notifying player " + e.getMessage());
            }
        }

        return output;
    }

    private static class Position {
        private int currentPosition;
        private int tries;
        private List<Boolean> visited;

        public Position() {
            this.currentPosition = 0;
            this.tries = 0;
            this.visited = new ArrayList<>(Arrays.asList(false, false, false, false, false));
        }

        public int getCurrentPosition() {
            return currentPosition;
        }

        public void setCurrentPosition(int currentPosition) {
            this.currentPosition = currentPosition;
        }

        public int getTries() {
            return tries;
        }

        public void setTries(int tries) {
            this.tries = tries;
        }

        public List<Boolean> getVisited() {
            return visited;
        }

        public void setVisited(List<Boolean> visited) {
            this.visited = visited;
        }
    }
}
