import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

// Map dupa titlul piesei
public class MyMap1{
    HashMap<Character, ArrayList<Mesaj>> map;
    Lock lock;

    public MyMap1(){
        this.map = new HashMap<>();
        this.lock = new ReentrantLock();
    }

    public void add(Mesaj value){
        lock.lock();
        try{
            Character key = value.titlu_piesa.charAt(0);
            if(map.containsKey(key)){
                ArrayList<Mesaj> list = map.get(key);
                list.add(value);
                map.put(key, list);
            }
            else{
                map.put(key, new ArrayList<>(List.of(value)));
            }
        }
        finally {
            lock.unlock();
        }
    }

    public int size(){
        lock.lock();
        try{
            int size = 0;
            for (Character key : map.keySet()) {
                size += map.get(key).size();
            }
            return size;
        } finally {
            lock.unlock();
        }
    }
}
