package com.socialNetwork.domain.validators;

import com.socialNetwork.domain.User;
import com.socialNetwork.exceptions.ValidationException;

import java.util.Objects;

public class UserValidator implements Validator<User> {

    private static final UserValidator instance = new UserValidator();

    private UserValidator() {}

    public static UserValidator getInstance() {
        return instance;
    }

    @Override
    public void validate(User entity) throws ValidationException {
        String err="";

        if (Objects.equals(entity.getFirstName(), "")) {
            err = err + "First name cannot be empty.\n";
        }
        if (Objects.equals(entity.getLastName(), "")) {
            err = err + "First name cannot be empty.\n";
        }

        if (err.length() > 0) {
            throw new ValidationException(err);
        }
    }
}
