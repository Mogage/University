import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class MyQueue<T> {
    private final int MAX = 50;
    private final Lock lock;
    private final Condition notFull;
    private final Condition notEmpty;
    private final AtomicInteger totalProducers;
    private final Queue<T> queue = new LinkedList<>();

    public MyQueue(AtomicInteger totalProducers) {
        this.totalProducers = totalProducers;
        this.lock = new ReentrantLock();
        this.notFull = lock.newCondition();
        this.notEmpty = lock.newCondition();
    }


    public void finish() {
        lock.lock();
        notEmpty.signalAll();
        notFull.signalAll();
        lock.unlock();
    }

    public void push(T element) throws InterruptedException {
        lock.lock();
        try {
            while (queue.size() == MAX) {
                notFull.await();
            }
            queue.add(element);
            notEmpty.signal();
        } finally {
            lock.unlock();
        }
    }

    public T pop() throws InterruptedException {
        T result;
        lock.lock();
        try {
            while (queue.isEmpty()) {
                if (totalProducers.get() == 0) {
                    return null;
                }
                notEmpty.await();
            }
            result = queue.remove();
            notFull.signal();
            return result;
        } finally {
            lock.unlock();
        }
    }

    public int size() {
        lock.lock();
        try{
            return queue.size();
        } finally {
            lock.unlock();
        }
    }

    public boolean hasData() {
        lock.lock();
        try{
            if (totalProducers.get() == 0) {
                return !queue.isEmpty();
            }
            return true;
        } finally {
            lock.unlock();
        }
    }
}