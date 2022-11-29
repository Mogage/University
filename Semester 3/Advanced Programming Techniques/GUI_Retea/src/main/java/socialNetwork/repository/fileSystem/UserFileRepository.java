package repository.fileSystem;

import domain.User;

public class UserFileRepository extends AbstractFileRepository<Long, User> {

    public UserFileRepository(String fileName) {
        super(fileName);
    }

    @Override
    protected User extractEntity(String line) {
        String[] words = line.split(",");
        User user = new User(words[1], words[2]);
        user.setId(Long.parseLong(words[0]));
        return user;
    }

    @Override
    protected String createStringFromEntity(User entity) {
        return entity.getId().toString() + ',' + entity.getFirstName() + ',' + entity.getLastName();
    }

}
