"""
    Author: Mogage Nicolae
    Creation date: 6 nov 2021
    Modul pentru tratarea exceptiilor
"""
class RentError(Exception):
    pass

class RentValidationError(RentError):
    pass

class RentRepositoryError(RentError):
    pass

class BookError(RentError):
    pass

class BookValidationError(BookError):
    pass

class BookRepositoryError(BookError):
    pass

class ClientError(RentError):
    pass

class ClientValidationError(ClientError):
    pass

class ClientRepositoryError(ClientError):
    pass

