import domain.User;
import repository.Repository;
import repository.UserDBRepository;

public class Main {
    public static void main(String[] args) {
        String url = "jdbc:postgresql://localhost:5432/academic";
        UserDBRepository userDBRepository = new UserDBRepository(url, "postgres", "nm13j4d25h");

        User user = new User("Selena", "Georgescu");
        userDBRepository.save(user);
        userDBRepository.findAll().forEach(System.out::println);
    }
}