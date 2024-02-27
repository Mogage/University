import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Main {
    public static void main(String[] args) {
        Lock lock = new ReentrantLock();
        Condition conditionReader = lock.newCondition();
        Condition conditionWorker = lock.newCondition();
        MyDictionary dictionary = new MyDictionary(lock, conditionReader, conditionWorker);
        int size = 100;
        MyWorkerThread thread1 = new MyWorkerThread(dictionary, size, "1", "debit");
        MyWorkerThread thread2 = new MyWorkerThread(dictionary, size, "2", "debit");
        MyWorkerThread thread3 = new MyWorkerThread(dictionary, size, "1", "credit");
        MyWorkerThread thread4 = new MyWorkerThread(dictionary, size, "2", "credit");
        MyReaderThread thread5 = new MyReaderThread(dictionary);

        IOHandler.emptyFile();

        thread5.start();
        thread1.start();
        thread2.start();
        thread3.start();
        thread4.start();

        try {
            thread1.join();
            thread2.join();
            thread3.join();
            thread4.join();
            thread5.join();
        } catch (InterruptedException interruptedException) {
            interruptedException.printStackTrace();
        }
    }
}