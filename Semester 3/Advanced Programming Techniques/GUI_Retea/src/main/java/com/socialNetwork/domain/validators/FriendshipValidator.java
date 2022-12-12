package com.socialNetwork.domain.validators;

import com.socialNetwork.domain.Friendship;
import com.socialNetwork.exceptions.ValidationException;

public class FriendshipValidator implements Validator<Friendship> {

    private static final FriendshipValidator instance = new FriendshipValidator();

    private FriendshipValidator() {
    }

    public static FriendshipValidator getInstance() {
        return instance;
    }

    @Override
    public void validate(Friendship entity) throws ValidationException {
        String err = "";

        if (entity.getIdUser1() < 0) {
            err += "Invalid first id for friendship.\n";
        }

        if (entity.getIdUser2() < 0) {
            err += "Invalid second id for friendship.\n";
        }

        if (err.length() > 0) {
            throw new ValidationException(err);
        }
    }
}
