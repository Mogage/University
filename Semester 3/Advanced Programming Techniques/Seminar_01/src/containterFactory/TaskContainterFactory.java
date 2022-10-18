package containterFactory;

import container.Container;
import utils.Constants;

public class TaskContainterFactory implements Factory {

    private static final TaskContainterFactory instance = new TaskContainterFactory();

    private TaskContainterFactory() {
    }

    public static TaskContainterFactory getInstance() {
        return instance;
    }

    public Container createContainer(Constants.Strategy strategy) {
        return Constants.strategyContainerMap.get(strategy);
    }
}
