"""
    Author: Mogage Nicolae
    Creation date: 16 nov 2021
    Modul pentru stocarea inchirierilor
"""
from domain.data import Rent, Books, Clients
from error.errors import RentRepositoryError

class RentRepository:
    """
        Clasa pentru stocarea entitatilor de tip inchiriere
    """
    def __init__(self):
        """
            Initializarea listei in care vor fi stocare inchirierile
        """
        self.__rent = []

    @property
    def rent(self):
        """
            Getter pentru lista cu entitati
        :return: list []
        """
        return self.__rent

    def add_rent(self, rent):
        """
            Adaugarea unei inchirieri in ordine crescatoare in memorie
        :param rent: inchiriere
        :raise RentRepositoryError: Daca aceasta inchiriere s-a mai efectuat
        """
        if rent in self.__rent:
            raise RentRepositoryError("Acesta carte este deja inchiriata de acest client.")

        index = 0
        numbers_of_rent = len(self.__rent)
        while index < numbers_of_rent and rent > self.__rent[index]:
            index = index + 1

        self.__rent.insert(index, rent)

    def delete_rent(self, id_book, id_client):
        """
            Stergerea / Returnarea de catre unui client cu id-ul id_client a cartii cu id_book
        :param id_book: id-ul unei carti - int
        :param id_client: id-ul unui client - int
        :raise RentRepositoryError: Daca clientul cu id-ul id_client nu a inchiriat nicio carte
        """
        index = 0
        number_of_rent = len(self.__rent)
        while index < number_of_rent:
            if self.__rent[index].client.id == id_client and self.__rent[index].book.id == id_book:
                self.__rent.pop(index)
                return
            index = index + 1

        raise RentRepositoryError("Acest client nu are inchiriata nicio carte in acest moment.")

    def print_all(self):
        """
            Afisarea tuturor inchirierilor efectuate
        """
        for _rent in self.__rent:
            print(str(_rent))

    def print_rent_by_client_id(self, id):
        """
            Afisarea tuturor inchirierilor efectuate de un client cu id-ul id
        :param id: id client - int
        :raise RentRepositoryError: Daca clientul cu id-ul id nu a inchiriat nicio carte
        """
        rented = False
        for _rent in self.__rent:
            if _rent.client.id == id:
                print(str(_rent))
                rented = True

        if not rented:
            raise RentRepositoryError("Acest client nu are inchiriata nicio carte in acest moment.")

    def check_rent_client_book(self, id_book, id_client):
        """
            Verifica daca clientul cu id-ul id_client a inchiriat cartea cu id_book
        :param id_book: id-ul unei carti - int
        :param id_client: id-ul unui client - int
        :raises RentRepositoryError: Daca acest client nu a inchiriat aceasta carte
        """
        for _rent in self.__rent:
            if _rent.client.id == id_client and _rent.book.id == id_book:
                return True

        raise RentRepositoryError("Acest client nu a inchiriat aceasta carte.")

    def check_rent_by_client_id(self, id):
        """
            Verifica daca un client cu id-ul id a efectuat cel putin o inchiriere
        :param id: id-ul unui client
        :return: True, daca exista o inchiriere efectuata de acest client
        :raise RentRepositoryError: Daca acest client nu a efectuat nicio inchiriere
        """
        for _rent in self.__rent:
            if _rent.client.id == id:
                return True

        raise RentRepositoryError("Acest client nu are inchiriata nicio carte in acest moment.")

    def __len__(self):
        """
            Lungimea unui repo este egala cu numarul de inchirieri efectuate
        :return: int - numarul de inchirieri efectuate
        """
        return len(self.__rent)

class RentFileRepository(RentRepository):
    """
        Clasa pentru stocarea in fisiere a inchirierilor
    """
    def __init__(self, file_name):
        """
            Initializarea datelor pentru stocarea in fisiere
        :param file_name: nume fisier - string
        """
        self.__file_name = file_name
        RentRepository.__init__(self)
        self.__load_from_file()

    @staticmethod
    def __create_rent_from_file(line):
        """
            Creeaza o inchiriere din string-ul line in care atributele sunt despartite prin ;
        :param line: o linie formatata din fisier - string
        :return: o inchiriere
        """
        fields = line.split(";")
        fields[7] = fields[7].removesuffix("\n")
        book = Books(int(fields[0]), fields[1], fields[2], fields[3])
        client = Clients(int(fields[4]), fields[5], int(fields[6]))
        rent = Rent(book, client, fields[7])
        return rent

    def __load_from_file(self):
        """
            Incarca toate datele din fisier
        """
        with open(self.__file_name) as rent_file:
            for line in rent_file:
                rent = self.__create_rent_from_file(line)
                RentRepository.add_rent(self, rent)

    @staticmethod
    def __create_book_line(book):
        """
            Formateaza o carte pentru a fi stocata in fisier
        :param book: o carte
        :return: un string cu atributele separate prin ;
        """
        return str(book.id) + ";" + book.title + ";" + book.description + ";" + book.author

    @staticmethod
    def __create_client_line(client):
        """
            Formateaza un client pentru a fi stocat in fisier
        :param client: un client
        :return: un string cu atributele separate prin ;
        """
        return str(client.id) + ";" + client.name + ";" + str(client.cnp)

    def __create_line(self, rent):
        """
            Formateaza o inchiriere pentru a fi stocata in fisier
        :param rent: o inchiriere
        :return: un string cu toate atributele separate prin ;
        """
        return self.__create_book_line(rent.book) + ";" + self.__create_client_line(rent.client) + ";" + rent.date

    @staticmethod
    def __add_to_file(line, rent_file):
        """
            Scrie in fisier inchirierea formatata pentru o linie
        :param line: inchiriere formatata sub forma de linie - string
        :param rent_file: fisierul in care se vor stoca
        """
        rent_file.write(line)
        rent_file.write("\n")

    def rewrite_file(self):
        """
            Scrie in fisier toate inchirierile efectuate
        """
        with open(self.__file_name, "w") as rent_file:
            for _rent in self.rent:
                line = self.__create_line(_rent)
                self.__add_to_file(line, rent_file)
