package utils;

import container.Container;
import container.QueueContainer;
import container.StackContainer;

import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

public class Constants {
    public static final DateTimeFormatter DATE_TIME_FORMATTER = DateTimeFormatter.ofPattern("dd-MM-yyyy hh:mm:ss");
    public static final DateTimeFormatter HOUR_FORMATTER = DateTimeFormatter.ofPattern("HH:mm:ss");

    public static final int TASKS_INITIAL_SIZE = 100;

    public enum Strategy {
        FIFO,
        LIFO
    }

    public static Map<Strategy, Container> strategyContainerMap = new HashMap<>() {{
        put(Strategy.FIFO, new StackContainer());
        put(Strategy.LIFO, new QueueContainer());
    }};
}