package com.socialNetwork.domain.validators;

import com.socialNetwork.domain.Message;
import com.socialNetwork.exceptions.ValidationException;

public class MessageValidator implements Validator<Message> {

    private static final MessageValidator instance = new MessageValidator();

    private MessageValidator() {
    }

    public static MessageValidator getInstance() {
        return instance;
    }

    @Override
    public void validate(Message entity) throws ValidationException {
        String errors = "";

        if (entity.getText().isEmpty()) {
            errors = "Message can't be empty.\n";
        }

        if (errors.length() > 0){
            throw new ValidationException(errors);
        }
    }
}
