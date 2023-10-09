package server.implementations;

import model.DTOFinishedGame;
import model.Game;
import model.Player;
import persistence.games.IGameRepository;
import persistence.players.IPlayerRepository;
import services.IObserver;
import services.IService;

import java.time.LocalTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class ServiceImplementation implements IService {
    IPlayerRepository playerRepository;
    IGameRepository gameRepository;
    private final Map<Integer, IObserver> loggedPlayers;
    private final Map<Integer, Game> games;
    private final Map<Integer, Integer> currentRows;
    private final Map<Integer, LocalTime> startTimes;

    public ServiceImplementation(IPlayerRepository playerRepository, IGameRepository gameRepository) {
        this.playerRepository = playerRepository;
        this.gameRepository = gameRepository;
        loggedPlayers = new ConcurrentHashMap<>();
        games = new ConcurrentHashMap<>();
        currentRows = new ConcurrentHashMap<>();
        startTimes = new ConcurrentHashMap<>();
    }

    @Override
    public synchronized Player login(Player player, IObserver client) throws Exception {
        Player playerToLogin = playerRepository.findByUsername(player.getUsername());
        if (playerToLogin == null) {
            throw new Exception("Authentication failed.");
        }
        if (loggedPlayers.get(playerToLogin.getId()) != null) {
            throw new Exception("Player already logged in.");
        }
        loggedPlayers.put(playerToLogin.getId(), client);

        List<String> positions = new ArrayList<>();
        Random random = new Random();

        positions.add("1 " + (random.nextInt(4) + 1));
        positions.add("2 " + (random.nextInt(4) + 1));
        positions.add("3 " + (random.nextInt(4) + 1));
        positions.add("4 " + (random.nextInt(4) + 1));
        String randomPosition = (random.nextInt(4) + 1) + " " + (random.nextInt(4) + 1);
        while (positions.contains(randomPosition)) {
            randomPosition = (random.nextInt(4) + 1) + " " + (random.nextInt(4) + 1);
        }
        positions.add(randomPosition);

        Game game = gameRepository.startGame(playerToLogin.getUsername(), positions);

        games.put(playerToLogin.getId(), game);
        currentRows.put(playerToLogin.getId(), 1);
        startTimes.put(playerToLogin.getId(), LocalTime.now());

        return playerToLogin;
    }

    @Override
    public synchronized void logout(Player player) {
        loggedPlayers.remove(player.getId());
    }

    @Override
    public synchronized Collection<Game> getScores() {
        Collection<Game> gamesList = gameRepository.getAll();
        for (Game game1 : games.values()) {
            gamesList.removeIf(game2 -> Objects.equals(game1.getId(), game2.getId()));
        }
        return gamesList;
    }

    @Override
    public DTOFinishedGame getFinishedGameInfo(int id) {
        Game game = games.get(id);
        DTOFinishedGame finishedGame = new DTOFinishedGame(game.getScore(), game.getPositions());

        game.setDurata((int) startTimes.get(id).until(LocalTime.now(), ChronoUnit.SECONDS));
        gameRepository.updateGame(game);

        games.remove(id);
        currentRows.remove(id);
        startTimes.remove(id);

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

        return finishedGame;
    }

    @Override
    public String move(int id, int row, int column) {
        Game game = games.get(id);
        int currentRow = currentRows.get(id);

        if (row != currentRow) {
            return "You are on the row " + currentRow;
        }

        String position = row + " " + column;
        for (String position1 : game.getProposedPositions()) {
            if (position.equals(position1)) {
                return "You already moved here.";
            }
        }

        game.getProposedPositions().add(position);

        String output = "";

        for (String position1 : game.getPositions()) {
            if (position.equals(position1)) {
                output = "There was a hole, you lost.";
                break;
            }
        }

        if (row == 4 && output.equals("")) {
            output = "Game won";
        }

        if (!output.equals("There was a hole, you lost.")) {
            currentRows.put(id, currentRow + 1);
            game.setScore(game.getScore() + row);
            gameRepository.updateGame(game);
            games.put(id, game);
        }

        return output;
    }
}
