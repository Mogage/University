package ati.utils;

import java.util.ArrayList;
import java.util.List;

public interface Observable {
    final List<Observer> observerList = new ArrayList<>();

    default void addObserver(Observer obs) {
        observerList.add(obs);
    }

    default void removeObserver(Observer obs) {
        observerList.remove(obs);
    }

    default void notifyObservers() {
        for(Observer observer : observerList) {
            observer.update();
        }
    }
}
