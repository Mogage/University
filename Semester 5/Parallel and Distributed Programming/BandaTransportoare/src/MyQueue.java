import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class MyQueue<T> {
    private final int max;
    private final Lock lock;
    private final Condition notFull;
    private final Condition notEmpty;
    private final AtomicInteger totalProducerTasks;
    private final Queue<T> queue = new LinkedList<>();

    public MyQueue(Lock lock, AtomicInteger totalProducersTasks, int max) {
        this.lock = lock;
        notFull = lock.newCondition();
        notEmpty = lock.newCondition();
        this.totalProducerTasks = totalProducersTasks;
        this.max = max;
    }

    public void push(T element, MyList<MyObject> list, int threadId) throws InterruptedException {
        lock.lock();
        try {
            while (queue.size() == max) {
                notFull.await();
            }
            queue.add(element);
            notEmpty.signal();
        } finally {
            list.add(new MyObject(threadId, queue.size(), "push"));
            lock.unlock();
        }
    }

    public void pop(MyList<MyObject> list, int threadId) throws InterruptedException {
        lock.lock();
        try {
            while (queue.isEmpty()) {
                if (totalProducerTasks.get() == 0) {
                    return;
                }
                notEmpty.await();
            }
            queue.remove();
            notFull.signal();
        } finally {
            list.add(new MyObject(threadId, queue.size(), "pop"));
            lock.unlock();
        }
    }

    public int size() {
        lock.lock();
        int toReturn = queue.size();
        lock.unlock();
        return toReturn;
    }

    public boolean hasData() {
        if (totalProducerTasks.get() == 0) {
            return !queue.isEmpty();
        }
        return true;
    }
}