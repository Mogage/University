package examen.utils.observer;

import java.util.ArrayList;
import java.util.List;

public interface Observable {
    List<Observer> observers = new ArrayList<>();

    default void addObserver(Observer obs) {
        observers.add(obs);
    }

    default void notifyObservers() {
        for(Observer observer : observers) {
            observer.update();
        }
    }
}
