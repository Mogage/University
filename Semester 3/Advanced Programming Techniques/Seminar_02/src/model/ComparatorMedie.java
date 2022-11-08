package model;

import java.util.Comparator;

public class ComparatorMedie implements Comparator<Student> {
    @Override
    public int compare(Student s1, Student s2) {
        int rounded1 = Math.round(s1.getMedia());
        int rounded2 = Math.round(s2.getMedia());

        return rounded1 - rounded2;
    }
}
