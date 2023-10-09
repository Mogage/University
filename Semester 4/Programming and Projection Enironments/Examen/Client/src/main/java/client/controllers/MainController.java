package client.controllers;

import javafx.beans.property.ReadOnlyObjectWrapper;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import model.DTOFinishedGame;
import model.Game;
import model.Player;
import services.IObserver;
import services.IService;

import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

public class MainController implements IObserver {
    @FXML
    public Label box11;
    @FXML
    public Label box21;
    @FXML
    public Label box31;
    @FXML
    public Label box41;
    @FXML
    public Label box12;
    @FXML
    public Label box22;
    @FXML
    public Label box32;
    @FXML
    public Label box42;
    @FXML
    public Label box13;
    @FXML
    public Label box23;
    @FXML
    public Label box33;
    @FXML
    public Label box43;
    @FXML
    public Label box14;
    @FXML
    public Label box24;
    @FXML
    public Label box34;
    @FXML
    public Label box44;
    @FXML
    public Label score;
    @FXML
    public Label holesPosition;
    @FXML
    public TableView<Game> scoresTable;
    @FXML
    public TableColumn<Game, String> usernameColumn;
    @FXML
    public TableColumn<Game, Integer> scoreColumn;
    @FXML
    public TableColumn<Game, Integer> gameLengthColumn;

    private final ObservableList<Game> scores = FXCollections.observableArrayList();

    private IService service;
    private Player loggedInPlayer;
    private boolean gameFinished = false;

    public void setService(IService service) {
        this.service = service;
    }

    public void setLoggedInPlayer(Player loggedInPlayer) {
        this.loggedInPlayer = loggedInPlayer;
    }

    private int compareFunction(Game game1, Game game2) {
        if (game1.getScore() == game2.getScore()) {
            return game2.getDurata() - game1.getDurata();
        }
        return game2.getScore() - game1.getScore();
    }

    public void initialise() {
        usernameColumn.setCellValueFactory(new PropertyValueFactory<>("player"));
        scoreColumn.setCellValueFactory(new PropertyValueFactory<>("score"));
        gameLengthColumn.setCellValueFactory(new PropertyValueFactory<>("durata"));
        try {
            scores.setAll(service.getScores());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        scoresTable.setItems(scores.sorted(this::compareFunction));
    }

    @Override
    public void gameFinished(Collection<Game> gamesList) {
        scores.setAll(gamesList);
        scoresTable.setItems(scores.sorted(this::compareFunction));
    }

    private String clickAction(int row, int column) {
        try {
            return service.move(loggedInPlayer.getId(), row, column);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private void updateHoles(int row, int column) {
        switch (row) {
            case 1 -> {
                switch (column) {
                    case 1 -> box11.setText("H");
                    case 2 -> box12.setText("H");
                    case 3 -> box13.setText("H");
                    case 4 -> box14.setText("H");
                }
            }
            case 2 -> {
                switch (column) {
                    case 1 -> box21.setText("H");
                    case 2 -> box22.setText("H");
                    case 3 -> box23.setText("H");
                    case 4 -> box24.setText("H");
                }
            }
            case 3 -> {
                switch (column) {
                    case 1 -> box31.setText("H");
                    case 2 -> box32.setText("H");
                    case 3 -> box33.setText("H");
                    case 4 -> box34.setText("H");
                }
            }
            case 4 -> {
                switch (column) {
                    case 1 -> box41.setText("H");
                    case 2 -> box42.setText("H");
                    case 3 -> box43.setText("H");
                    case 4 -> box44.setText("H");
                }
            }
        }
    }

    private void finishGame() {
        DTOFinishedGame finishedGame;
        try {
            finishedGame = service.getFinishedGameInfo(loggedInPlayer.getId());
        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            alert.show();
            return;
        }
        score.setText("Score: " + finishedGame.getScore());
        List<String> positions = finishedGame.getHolesPosition();
        holesPosition.setText("Holes position: " + positions);
        for (String position : positions) {
            String[] coordinates = position.split(" ");
            int row = Integer.parseInt(coordinates[0]);
            int column = Integer.parseInt(coordinates[1]);
            updateHoles(row, column);
        }
    }

    private boolean manageClick(String output) {
        if (Objects.equals(output, "There was a hole, you lost.") || Objects.equals(output, "Game won")) {
            gameFinished = true;
            finishGame();
            return Objects.equals(output, "Game won");
        }
        if (!Objects.equals(output, "")) {
            Alert alert = new Alert(Alert.AlertType.INFORMATION, output, ButtonType.OK);
            alert.show();
            return false;
        }
        return true;
    }

    public void box11Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(1, 1);
        if (manageClick(output)) {
            box11.setText("F");
        }
    }

    public void box21Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(2, 1);
        if (manageClick(output)) {
            box21.setText("F");
        }
    }

    public void box31Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(3, 1);
        if (manageClick(output)) {
            box31.setText("F");
        }
    }

    public void box41Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(4, 1);
        if (manageClick(output)) {
            box41.setText("F");
        }
    }

    public void box12Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(1, 2);
        if (manageClick(output)) {
            box12.setText("F");
        }
    }

    public void box22Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(2, 2);
        if (manageClick(output)) {
            box22.setText("F");
        }
    }

    public void box32Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(3, 2);
        if (manageClick(output)) {
            box32.setText("F");
        }
    }

    public void box42Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(4, 2);
        if (manageClick(output)) {
            box42.setText("F");
        }
    }

    public void box13Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(1, 3);
        if (manageClick(output)) {
            box13.setText("F");
        }
    }

    public void box23Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(2, 3);
        if (manageClick(output)) {
            box23.setText("F");
        }
    }

    public void box33Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(3, 3);
        if (manageClick(output)) {
            box33.setText("F");
        }
    }

    public void box43Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(4, 3);
        if (manageClick(output)) {
            box43.setText("F");
        }
    }

    public void box14Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(1, 4);
        if (manageClick(output)) {
            box14.setText("F");
        }
    }

    public void box24Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(2, 4);
        if (manageClick(output)) {
            box24.setText("F");
        }
    }

    public void box34Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(3, 4);
        if (manageClick(output)) {
            box34.setText("F");
        }
    }

    public void box44Click() {
        if (gameFinished) {
            return;
        }
        String output = clickAction(4, 4);
        if (manageClick(output)) {
            box44.setText("F");
        }
    }
}
