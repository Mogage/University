package container;

import model.Task;

public class QueueContainer extends AbstractContainer {
    public QueueContainer() {
        super();
    }

    @Override
    public Task remove() {
        if (isEmpty()) {
            return null;
        }
        Task removed = tasks[0];
        for (int count = 0; count < tasks.length - 1; count++) {
            tasks[count] = tasks[count + 1];
        }
        size--;
        return removed;
    }
}