class EventError(Exception):
    pass


class RepositoryError(EventError):
    pass


class ValidatorError(EventError):
    pass
