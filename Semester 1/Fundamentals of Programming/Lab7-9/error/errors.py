"""
    Author: Mogage Nicolae
    Creation date: 6 nov 2021
    Modul pentru tratarea exceptiilor
"""
class BookValidationError(Exception):
    pass

class BookRepositoryError(Exception):
    pass

class ClientValidationError(Exception):
    pass

class ClientRepositoryError(Exception):
    pass
