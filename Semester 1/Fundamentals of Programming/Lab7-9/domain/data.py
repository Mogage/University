"""
    Author: Mogage Nicolae
    Creation date: 6 nov 2021
    Modul pentru entitati
"""

class Books:
    """
        Clasa care gestioneaza o entitate de tip carte
    """

    def __init__(self, id_book, book_title, book_description, book_author):
        """
            Functie de initializare a datelor unei carti
        :param id_book: int > 0
        :param book_title: string nevid
        :param book_description: string nevid
        :param book_author: string nevid
        """
        self.__id = id_book
        self.__title = book_title
        self.__description = book_description
        self.__author = book_author

    @property
    def id(self):
        """
            Functie de getter pentru id
        :return: __id
        """
        return self.__id

    @property
    def title(self):
        """
            Functie de getter pentru title
        :return: __title
        """
        return self.__title

    @title.setter
    def title(self, book_title):
        """
            Functie de setter pentru title
        :param book_title: string nevid
        :return: -
        """
        self.__title = book_title

    @property
    def description(self):
        """
            Functie de getter pentru description
        :return: __description
        """
        return self.__description

    @description.setter
    def description(self, book_description):
        """
            Functie de setter pentru description
        :param book_description: string nevid
        :return: -
        """
        self.__description = book_description

    @property
    def author(self):
        """
            Functie de getter pentru author
        :return: __author
        """
        return self.__author

    @author.setter
    def author(self, book_author):
        """
            Functie de setter pentru author
        :param book_author: string nevid
        :return: -
        """
        self.__author = book_author

    def __gt__(self, other):
        """
            Spunem ca o carte self este mai mare decat alta other cand id-ul lui self e mai mare decat id-ul lui other
        :param other: o carte
        :return: True, daca self.__id > other.id
                 False, altfel
        """
        return self.__id > other.id

    def __eq__(self, other):
        """
            Spunem ca 2 carti sunt egale daca au acelasi id
        :param other: o carte
        :return: True, daca self.__id == other.id
                 False, altfel
        """
        return self.__id == other.id

    def __str__(self):
        """
            Se pregateste formatul unei carti pentru afisare
        :return: carte in format:
                [id]Titlu: title
                Descriere: description
                Autor: author
        """
        return "[" + str(
            self.__id) + "]Titlu: " + self.__title + "\nDescriere: " + self.__description + "\nAutor: " + self.__author

class Clients:
    """
        Clasa care gestioneaza entitatea de tip client
    """

    def __init__(self, id_client, name, cnp):
        """
            Functie de initializare a datelor unui client
        :param id_client: int > 0
        :param name: string nevid
        :param cnp: int cu exact 13 cifre
        """
        self.__id = id_client
        self.__name = name
        self.__cnp = cnp

    @property
    def id(self):
        """
            Functie de getter pentru id
        :return: __id
        """
        return self.__id

    @property
    def name(self):
        """
            Functie de getter pentru nume
        :return: __name
        """
        return self.__name

    @name.setter
    def name(self, name):
        """
            Functie de setter pentru nume
        :param name: string nevid
        :return: -
        """
        self.__name = name

    @property
    def cnp(self):
        """
            Functie de getter pentru cnp
        :return: __cnp
        """
        return self.__cnp

    def __gt__(self, other):
        """
            Spunem ca un client self este mai mare decat altul other cand id-ul lui self e mai mare decat id-ul lui other
        :param other: un client
        :return: True, daca self.__id > other.__id
                 False, altfel
        """
        return self.__id > other.id

    def __eq__(self, other):
        """
            Spunem ca 2 clienti sunt egali daca au acelasi id sau daca au acelasi cnp
        :param other: un client
        :return: True, daca self.__id == other.id sau self.__cnp == other.cnp
                 False, altfel
        """
        return self.__id == other.id or self.__cnp == other.cnp

    def __str__(self):
        """
            Se pregateste formatul unui client pentru afisare
        :return: client in format: [id]Nume: name. CNP: cnp
        """
        return "[" + str(self.__id) + "]Nume: " + self.__name + ". CNP: " + str(self.__cnp)
