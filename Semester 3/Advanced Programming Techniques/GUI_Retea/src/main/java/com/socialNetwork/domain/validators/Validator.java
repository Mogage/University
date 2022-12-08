package com.socialNetwork.domain.validators;

import com.socialNetwork.exceptions.ValidationException;

public interface Validator<T> {
    /**
     * Validates an entity
     * @param entity - entity to validate
     * @throws ValidationException - in case the entity is not valid
     */
    void validate(T entity) throws ValidationException;
}