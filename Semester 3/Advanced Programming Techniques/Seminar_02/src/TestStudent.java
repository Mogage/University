import model.Student;
import model.validators.StudentValidator;
import model.validators.Validator;
import repository.StudentsFileRepository;

public class TestStudent {
    public static void main(String[] args) {
        Validator<Student> validator = new StudentValidator();
        StudentsFileRepository studentsFileRepository = new StudentsFileRepository(validator, "data\\studentsData.csv");
        studentsFileRepository.findAll().forEach(System.out::println);
    }
}
