import domain.Friendship;
import domain.User;
import domain.validators.FriendshipValidator;
import domain.validators.UserValidator;
import domain.validators.Validator;
import network.Network;
import network.UserNetwork;
import repository.InMemoryRepository;
import repository.Repository;
import service.Service;
import service.UserService;
import userInterface.UI;

public class Main {
    public static void main(String[] args) {
        Validator<User> userValidator = UserValidator.getInstance();
        Validator<Friendship> friendshipValidator = FriendshipValidator.getInstance();
        Repository<Long, User> inMemoryRepository = new InMemoryRepository<>();
        Network<User> userNetwork = new UserNetwork();
        Service<User> userService = new UserService(userValidator, friendshipValidator, inMemoryRepository, userNetwork);
        UI userInterface = new UI(userService);

        userInterface.run();
    }
}