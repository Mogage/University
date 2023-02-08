package anar.utils;

import java.util.ArrayList;
import java.util.List;

public interface Observable {

    List<Observer> observers = new ArrayList<>();

    default void addObserver(Observer obs) {
        observers.add(obs);
    }

    default void removeObserver(Observer obs) {
        observers.remove(obs);
    }

    default void notifyObservers() {
        for(Observer observer : observers) {
            observer.update();
        }
    }
}
