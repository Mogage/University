import domain.Friendship;
import domain.User;
import domain.validators.FriendshipValidator;
import domain.validators.UserValidator;
import domain.validators.Validator;
import network.Network;
import network.MainNetwork;
import repository.FriendshipFileRepository;
import repository.Repository;
import repository.UserFileRepository;
import service.Service;
import service.MainService;
import userInterface.UI;

public class Main {
    public static void main(String[] args) {
        Validator<User> userValidator = UserValidator.getInstance();
        Validator<Friendship> friendshipValidator = FriendshipValidator.getInstance();
        //Repository<Long, User> inMemoryRepository = new InMemoryRepository<>();
        Repository<Long, User> userRepository = new UserFileRepository("data\\usersData.csv");
        Repository<Long, Friendship> friendshipRepository = new FriendshipFileRepository("data\\friendshipsData.csv");
        Network network = new MainNetwork();
        Service service = new MainService(userValidator, friendshipValidator, userRepository, friendshipRepository, network);
        UI userInterface = new UI(service);

        userInterface.run();
    }
}