package taskRunner;

import container.Container;
import containterFactory.TaskContainterFactory;
import model.Task;
import utils.Constants;

public class StrategyTaskRunner implements TaskRunner {
    private final Container container;

    public StrategyTaskRunner(Constants.Strategy strategy) {
        TaskContainterFactory factory = TaskContainterFactory.getInstance();
        container = factory.createContainer(strategy);
    }

    public void executeOneTask() {
        if (container.isEmpty()) {
            return;
        }
        Task removed = container.remove();
        removed.execute();
    }

    public void executeAll() {
        while (!container.isEmpty()) {
            executeOneTask();
        }
    }

    public void addTask(Task t) {
        container.add(t);
    }

    public boolean hasTask() {
        return !container.isEmpty();
    }
}
