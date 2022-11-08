package repository;

import model.Student;
import model.validators.ValidationException;
import model.validators.Validator;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class StudentsFileRepository extends InMemoryRepository<Long, Student> {
    private final String fileName;

    public StudentsFileRepository(Validator validator, String fileName) {
        super(validator);
        this.fileName = fileName;
        loadData();
    }

    private void loadData() {
        Path path = Paths.get(fileName);
        try {
            List<String> lines = Files.readAllLines(path);
            lines.forEach(line -> {
                String[] words = line.split(";");
                Student student = new Student(words[1], Float.parseFloat(words[2]));
                student.setId(Long.parseLong(words[0]));
                try {
                    super.save(student);
                } catch (ValidationException e) {
                    throw new RuntimeException(e);
                }
            });

        } catch (IOException e) {
            System.out.println("Eroare la citirea fisierului!\n");
            e.printStackTrace();
        }
    }
}







