import model.Student;

import customMap.MyMap;

import java.util.*;

public class TestRunner {
    static Student s2 = new Student("Ana", 8.5f);
    static Student s3 = new Student("Dan", 4.5f);
    static Student s1 = new Student("Dan", 4.5f);
    static Student s4 = new Student("Bianca", 5.5f);
    static Student s5 = new Student("Serban", 9.5f);

    private static void runHashSet() {
        Set<Student> studentSet = new HashSet<>(Arrays.asList(s1, s2, s3));

        System.out.println("Hash set:");
        for (Student student : studentSet) {
            System.out.println(student);
        }
    }

    private static void runTreeSet() {
        TreeSet<Student> studentTreeSet = new TreeSet<>(new Comparator<Student>() {
            @Override
            public int compare(Student student1, Student student2) {
                return student1.getName().compareTo(student2.getName());
            }
        });
        studentTreeSet.addAll(Arrays.asList(s1, s2, s3, s4, s5));

        System.out.println("Tree set:");
        for (Student student : studentTreeSet) {
            System.out.println(student);
        }
    }

    private static void addStudents(Map<String, Student> MapStudent) {
        MapStudent.put(s1.getName(), s1);
        MapStudent.put(s2.getName(), s2);
        MapStudent.put(s3.getName(), s3);
        MapStudent.put(s4.getName(), s4);
        MapStudent.put(s5.getName(), s5);

        for (Map.Entry<String, Student> pair : MapStudent.entrySet()) {
            System.out.println("Key: " + pair.getKey() + ", Val: " + pair.getValue());
        }
    }

    private static void runHashMap() {
        System.out.println("HashMap");
        Map<String, Student> hashMapStudent = new HashMap<>();
        addStudents(hashMapStudent);
    }

    private static void runTreeMap() {
        System.out.println("TreeMap");
        Map<String, Student> treeMapStudent = new TreeMap<>();
        addStudents(treeMapStudent);
    }

    private static void runMyMap() {
        System.out.println("MyMap");
        MyMap map = new MyMap();
        map.add(s1);
        map.add(s2);
        map.add(s3);

        map.printAll();
    }



    public static void run() {
        runHashSet();
        runTreeSet();
        runHashMap();
        runTreeMap();
        runMyMap();
    }
}
