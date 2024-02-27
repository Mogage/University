import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class MyReaderThread extends Thread {
    private final MyDictionary dictionary;

    public MyReaderThread(MyDictionary dictionary) {
        this.dictionary = dictionary;
    }

    @Override
    public void run() {
        IOHandler.writeText("Reader thread started\n");
        while (true) {
            dictionary.lock.lock();
            while (dictionary.totalTransitions.get() < 50) {
                try {
                    dictionary.conditionReader.await();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            Iterator<Map.Entry<String, HashMap<String, Integer>>> iterator = dictionary.iterator();
            while (iterator.hasNext()) {
                Map.Entry<String, HashMap<String, Integer>> entry = iterator.next();
                IOHandler.writeText(entry.getKey() + " : " + entry.getValue() + "\n");
            }
            dictionary.totalTransitions.set(0);
            dictionary.conditionWorker.signalAll();
            dictionary.lock.unlock();
        }
    }
}
