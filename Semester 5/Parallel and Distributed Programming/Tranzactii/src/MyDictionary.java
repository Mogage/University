import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class MyDictionary {
    private final HashMap<String, HashMap<String, Integer>> dictionary;
    public Lock lock;
    public Condition conditionReader;
    public Condition conditionWorker;
    public AtomicInteger totalTransitions;
    public AtomicBoolean isShown;

    public MyDictionary(Lock lock, Condition conditionReader, Condition conditionWorker) {
        dictionary = new HashMap<>();
        this.lock = lock;
        this.conditionReader = conditionReader;
        this.conditionWorker = conditionWorker;
        totalTransitions = new AtomicInteger(0);
        isShown = new AtomicBoolean(false);
    }

    public synchronized boolean verifyKey(String key) {
        return dictionary.containsKey(key);
    }

    public synchronized void insertKey(String key) {
        dictionary.put(key, new HashMap<>());
    }

    public void pushValue(String key, String valueKey, Integer value) {
        lock.lock();
        while (totalTransitions.get() == 50) {
            try {
                if (!isShown.get()) {
                    isShown.set(true);
                    conditionReader.signal();

                    conditionWorker.await();
                    HashMap<String, Integer> innerMap = dictionary.get(key);
                    if (innerMap.containsKey(valueKey)) {
                        innerMap.put(valueKey, innerMap.get(valueKey) + value);
                    } else {
                        innerMap.put(valueKey, value);
                    }
                } else {
                    conditionWorker.await();
                }
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        if (!isShown.get()) {
            HashMap<String, Integer> innerMap = dictionary.get(key);
            if (innerMap.containsKey(valueKey)) {
                innerMap.put(valueKey, innerMap.get(valueKey) + value);
            } else {
                innerMap.put(valueKey, value);
            }
            totalTransitions.incrementAndGet();
        }
        isShown.set(false);
        IOHandler.writeText("Pushing value for " + key + " with operation type " + valueKey + " and value " + value + "\n");
        lock.unlock();
    }

    public synchronized void deleteValue(String key, String valueKey, Integer value) {
        HashMap<String, Integer> innerMap = dictionary.get(key);
        innerMap.remove(valueKey);
    }

    public Iterator<Map.Entry<String, HashMap<String, Integer>>> iterator() {
        return dictionary.entrySet().iterator();
    }
}
