package client.controllers;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.cell.PropertyValueFactory;
import model.DtoInitialise;
import model.Game;
import model.Player;
import services.IObserver;
import services.IService;

import java.util.Collection;
import java.util.Comparator;
import java.util.Objects;

public class MainController implements IObserver {
    @FXML
    public Label box1;
    @FXML
    public Label box2;
    @FXML
    public Label box3;
    @FXML
    public Label box4;
    @FXML
    public Label box5;
    @FXML
    public Label currentPosition;
    @FXML
    public Label finalScore;
    @FXML
    public TableView<Game> scoresTable;
    @FXML
    public TableColumn<Game, String> usernameColumn;
    @FXML
    public TableColumn<Game, String> dateColumn;
    @FXML
    public TableColumn<Game, String> timeColumn;
    @FXML
    public TableColumn<Game, Integer> scoreColumn;

    private final ObservableList<Game> scores = FXCollections.observableArrayList();

    private IService service;
    private int tries = 0;
    private Player loggedInPlayer;
    private boolean gameFinished = false;

    public void setService(IService service) {
        this.service = service;
    }

    public void setLoggedInPlayer(Player loggedInPlayer) {
        this.loggedInPlayer = loggedInPlayer;
    }

    public void initialise() {
        usernameColumn.setCellValueFactory(new PropertyValueFactory<>("player"));
        dateColumn.setCellValueFactory(new PropertyValueFactory<>("date"));
        timeColumn.setCellValueFactory(new PropertyValueFactory<>("time"));
        scoreColumn.setCellValueFactory(new PropertyValueFactory<>("score"));
        DtoInitialise dtoInitialise;
        try {
            dtoInitialise = service.initialise(loggedInPlayer.getId());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        box1.setText(String.valueOf(dtoInitialise.getConfiguration().getValue1()));
        box2.setText(String.valueOf(dtoInitialise.getConfiguration().getValue2()));
        box3.setText(String.valueOf(dtoInitialise.getConfiguration().getValue3()));
        box4.setText(String.valueOf(dtoInitialise.getConfiguration().getValue4()));
        box5.setText(String.valueOf(dtoInitialise.getConfiguration().getValue5()));
        currentPosition.setText("Current position: 1");
        scores.setAll(dtoInitialise.getScores());
        scoresTable.setItems(scores.sorted(Comparator.comparingInt(Game::getScore)));
    }

    public void moveAction() {
        if (gameFinished) {
            return;
        }

        try {
            int newPosition = service.move(loggedInPlayer.getId());
            currentPosition.setText("Current position: " + newPosition);
            tries++;
            if (tries == 3) {
                gameFinished = true;
                finalScore.setText("Final score: " + service.getScore(loggedInPlayer.getId()));
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void gameFinished(Collection<Game> gamesList) {
        scores.setAll(gamesList);
        scoresTable.setItems(scores.sorted(Comparator.comparingInt(Game::getScore)));
    }
}
