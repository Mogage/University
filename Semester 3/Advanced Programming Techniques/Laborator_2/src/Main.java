import domain.Friendship;
import domain.User;
import domain.validators.FriendshipValidator;
import domain.validators.UserValidator;
import domain.validators.Validator;
import network.Network;
import network.MainNetwork;
import repository.databaseSystem.FriendshipDBRepository;
import repository.databaseSystem.UserDBRepository;
import repository.fileSystem.UserFileRepository;
import repository.fileSystem.FriendshipFileRepository;
import repository.Repository;
import service.Service;
import service.MainService;
import userInterface.UI;

public class Main {
    public static void main(String[] args) {
        String url = "jdbc:postgresql://localhost:5432/laborator_2";
        String userName = "postgres";
        String password = "nm13j4d25h";

        Validator<User> userValidator = UserValidator.getInstance();
        Validator<Friendship> friendshipValidator = FriendshipValidator.getInstance();
        //Repository<Long, User> userRepository = new UserFileRepository("data\\usersData.csv");
        //Repository<Long, Friendship> friendshipRepository = new FriendshipFileRepository("data\\friendshipsData.csv");

        Repository<Long, User> userRepository = new UserDBRepository(url, userName, password);
        Repository<Long, Friendship> friendshipRepository = new FriendshipDBRepository(url, userName, password);

        Network network = new MainNetwork();
        Service service = new MainService(userValidator, friendshipValidator, userRepository, friendshipRepository, network);
        UI userInterface = new UI(service);

        userInterface.runMain();
    }
}