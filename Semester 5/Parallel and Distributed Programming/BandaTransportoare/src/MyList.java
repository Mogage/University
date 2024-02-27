import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class MyList<T> {
    public Node<?> head;
    public Node<?> tail;
    public final Lock lock;

    public MyList(Lock lock) {
        this.lock = lock;
        head = null;
        tail = null;
    }

    public void add(T nodeData) {
        lock.lock();
        Node<T> node = new Node<>(nodeData, null, null);
        if (head == null) {
            head = node;
            tail = node;
            lock.unlock();
            return;
        }
        tail.next = node;
        node.previous = tail;
        tail = node;

        lock.unlock();
    }

    public void delete() {
        lock.lock();

        if (head == null) {
            lock.unlock();
            return;
        }
        if (head.next == null) {
            head = null;
            tail = null;
            lock.unlock();
            return;
        }
        head = head.next;
        head.previous = null;
        lock.unlock();
    }

    public void printList() {
        lock.lock();
        Node<?> current = head;
        while (current != null) {
            System.out.println(current.getData());
            current = current.next;
        }
        lock.unlock();
    }
}
