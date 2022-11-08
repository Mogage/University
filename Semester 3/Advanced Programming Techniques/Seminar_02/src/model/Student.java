package model;

import java.util.Objects;

public class Student extends Entity<Long> implements Comparable<Student> {
    private String name;

    private float media;

    public Student(String name, float media) {
        this.name = name;
        this.media = media;
    }

    public String getName() {
        return name;
    }

    public float getMedia() {
        return media;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setMedia(float media) {
        this.media = media;
    }

    public int compareTo(Student student) {
        return Math.round(this.media - student.media);
    }

    @Override
    public String toString() {
        return name + " " + media;
    }

    @Override
    public int hashCode() {
        return Objects.hash(getMedia(), getName());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true; // verifica daca e acelasi referinta in memorie
        if (o == null || getClass() != o.getClass()) return false;
        Student student = (Student) o;
        return Float.compare(student.media, media) == 0 && Objects.equals(name, student.name);
    }
}
