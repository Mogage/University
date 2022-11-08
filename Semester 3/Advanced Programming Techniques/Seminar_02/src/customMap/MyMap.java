package customMap;

import model.Student;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class MyMap {
    private Map<Integer, List<Student>> map;

    public MyMap() {
        this.map = new TreeMap<>();
    }

    public int[] getKeys() {
        int[] keys = new int[map.size()];
        int n = 0;
        for (int key : map.keySet()) {
            keys[n] = key;
            n++;
        }
        return keys;
    }

    public void printAll() {
        int[] keys = getKeys();
        for (int key : keys) {
            List<Student> studentList = map.get(key);
            for (Student student : studentList) {
                System.out.println(student);
            }
        }
    }

    public void add(Student s) {
        float media = s.getMedia();
        int rounded = Math.round(media);
        List<Student> list = map.get(rounded);
        if (list == null) {
            list = new ArrayList<>();
            map.put(rounded, list);
        }
        list.add(s);
    }
}
