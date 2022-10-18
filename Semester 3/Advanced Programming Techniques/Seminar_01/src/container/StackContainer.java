package container;

import model.Task;

public class StackContainer extends AbstractContainer {
    public StackContainer() {
        super();
    }

    @Override
    public Task remove() {
        if (isEmpty()) {
            return null;
        }
        size--;
        return tasks[size];
    }
}
