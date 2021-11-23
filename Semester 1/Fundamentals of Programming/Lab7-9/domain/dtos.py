"""
    Author: Mogage Nicolae
    Creation date: 23 nov 2021
    Modul pentru transferul datelor
"""

class RentDto:
    """
        Clasa pentru gestiunea entitatii de tip inchiriere
    """
    def __init__(self, id_book, id_client, date):
        """
            Initalizarea datelor unei inchirieri
        :param id_book: id-ul unei carti
        :param id_client: id-ul unui client
        :param date: data in formatul dd/mm/yyyy
        """
        self.__id_book = id_book
        self.__id_client = id_client
        self.__date = date
        self.__book = None
        self.__client = None

    @property
    def id_book(self):
        """
            Getter pentru id-ul unei carti
        :return: int - id-ul unei carti
        """
        return self.__id_book

    @property
    def id_client(self):
        """
            Getter pentru id-ul unui client
        :return: int - id-ul unui client
        """
        return self.__id_client

    @property
    def date(self):
        """
            Getter pentru data inchirierii
        :return: date - data in formatul dd/mm/yyyy
        """
        return self.__date

    @property
    def book(self):
        """
            Getter pentru o carte
        :return: carte
        """
        return self.__book

    @book.setter
    def book(self, book):
        """
            Setter pentru o carte
        :param book: carte
        """
        self.__book = book

    @property
    def client(self):
        """
            Getter pentru un client
        :return: client
        """
        return self.__client

    @client.setter
    def client(self, client):
        """
            Setter pentru un client
        :param client: client
        """
        self.__client = client

    def __gt__(self, other):
        """
            o inchiriere self este mai mare decat una other daca id-ul clientului self e mai mare decat cel a lui other \
            sau in caz de egalitate, daca id-ul cartii self e mai mare decat id-ul cartii other
        :param other: inchiriere
        :return: True, daca inchirierea self > other
                 False, altfel
        """
        if self.__id_client != other.id_client:
            return self.__id_client > other.id_client

        return self.__id_book > other.id_book

    def __eq__(self, other):
        """
            2 inchirieri sunt egale daca au client cu acelasi id si carte cu acelasi id
        :param other: inchiriere
        :return: True, daca 2 inchiriere sunt egale
                 False, altfel
        """
        return self.__id_book == other.id_book and self.__id_client == other.id_client

    def __str__(self):
        """
            Pregateste o inchiriere pentru afisare
        :return: inchiriere in format "Clientul 'name' a inchiriat cartea 'titlu_carte' scrisa de 'autor_carte' \
                 la data de date"
        """
        return "Clientul '" + self.__client.name + "' a inchiriat cartea '" + self.__book.title + "' scrisa de '" + \
               self.__book.author + "' la data de " + str(self.__date)
