"""
    Author: Mogage Nicolae
    Creation date: 6 nov 2021
    Modul pentru validarea entitatilor si a atributelor lor
"""
from error.errors import BookValidationError, ClientValidationError, RentValidationError


class _MutualValidator:
    """
        Clasa pentru metode comune
    """
    @staticmethod
    def validate_id(id):
        """
            Verifica daca id-ul unei carti este valid
        :param id: int
        :return: True, if id > 0
                 False, else
        """
        return id > 0

    @staticmethod
    def validate_string(string):
        """
            Verifica daca un string nu este gol
        :param string: string
        :return: True, if string nonempty
                 False, else
        """
        return string != ""

class BooksValidator(_MutualValidator):
    """
        Clasa pentru validarea entitatii de tip carte si a atributelor ei
    """
    def __init__(self):
        """

        """
        _MutualValidator.__init__(self)

    @staticmethod
    def validate_id(id):
        """
            Verifica daca id-ul unei carti este valid
        :param id: int
        :return: -
        :raise: Id-ul introdus este invalid. if int <= 0
        """
        if not _MutualValidator.validate_id(id):
            raise BookValidationError("Id-ul introdus este invalid.")

    def validate_title(self, title):
        """
            Verifica daca titlul unei carti este valid
        :param title: string
        :return: -
        :raise: Titlul introdus este invalid. if title is empty
        """
        if not self.validate_string(title):
            raise BookValidationError("Titlul introdus este invalid.")

    def validate_description(self, description):
        """
            Verifica daca descrierea unei carti este valida
        :param description: string
        :return: -
        :raise: Descrierea introdusa este invalida. if description is empty
        """
        if not self.validate_string(description):
            raise BookValidationError("Descrierea introdusa este invalida.")

    def validate_author(self, author):
        """
            Verifica daca autorul unei carti este valid
        :param author: string
        :return: -
        :raise: Autorul introdus este invalid. if author is empty
        """
        if not self.validate_string(author):
            raise BookValidationError("Autorul introdus este invalid.")

    def validate_book(self, book):
        """
            Verifica daca o carte este valida, adica daca atributele ei sunt valide
        :param book: o carte
        :return: -
        :raise: string cu toate exceptiile daca s-a gasit cel putin un atribut invalid
        """
        errors = ""
        if book.id < 1:
            errors += "Id-ul introdus este invalid.\n"
        if not self.validate_string(book.title):
            errors += "Titlul introdus este invalid.\n"
        if not self.validate_string(book.description):
            errors += "Descrierea introdusa este invalida.\n"
        if not self.validate_string(book.author):
            errors += "Autorul introdus este invalid.\n"
        if self.validate_string(errors):
            raise BookValidationError(errors)

class ClientsValidator(_MutualValidator):
    """
        Clasa pentru validare unei entitati de tip client si atributele lui
    """
    def __init__(self):
        """

        """
        _MutualValidator.__init__(self)

    @staticmethod
    def __check_cnp(cnp):
        """
            Verifica daca un cnp este valid
        :param cnp: int
        :return: True, if number of digits in cnp is 13
                 False, else
        """
        return len(str(cnp)) == 13

    @staticmethod
    def validate_id(id):
        """
            Verifica daca id-ul unui client este valid
        :param id: int
        :return: -
        :raise: Id-ul introdus este invalid. if id <= 0
        """
        if not _MutualValidator.validate_id(id):
            raise ClientValidationError("Id-ul introdus este invalid.")

    def validate_prefix(self, prefix):
        """
            Verifica daca prefixul este valid
        :param prefix: string
        ":raise: Prefixul introdus este invalid. if prefix is empty
        """
        if not self.validate_string(prefix):
            raise ClientValidationError("Prefixul introdus este invalid.")

    def validate_name(self, name):
        """
            Verifica daca numele unui client este valid
        :param name: string
        :return:-
        :raise: Numele introdus este invalid. if string is empty
        """
        if not self.validate_string(name):
            raise ClientValidationError("Numele introdus este invalid.")

    def validate_cnp(self, cnp):
        """
            Verifica daca cnp-ul unei persoane este valid
        :param cnp: int
        :return: -
        :raise: Cnp-ul introdus este invalid. if cnp is not valid
        """
        if not self.__check_cnp(cnp):
            raise ClientValidationError("Cnp-ul introdus este invalid.")

    def validate_client(self, client):
        """
            Verifica daca un client este valid, adica daca atributele lui sunt valide
        :param client: un client
        :return: -
        :raise: string cu toate exceptiile daca s-a gasit cel putin un atribut invalid
        """
        errors = ""
        if client.id < 1:
            errors += "Id-ul introdus este invalid.\n"
        if not self.validate_string(client.name):
            errors += "Numele introdus este invalid.\n"
        if not self.__check_cnp(client.cnp):
            errors += "Cnp-ul introdus este invalid.\n"

        if self.validate_string(errors):
            raise ClientValidationError(errors)

class RentValidator(_MutualValidator):
    """
        Clasa pentru validarea entitatii de tip inchiriere si a atributelor ei
    """
    def __init__(self):
        """

        """
        _MutualValidator.__init__(self)

    @staticmethod
    def validate_id(id):
        if not _MutualValidator.validate_id(id):
            raise RentValidationError("Id-ul introdus este invalid.")
