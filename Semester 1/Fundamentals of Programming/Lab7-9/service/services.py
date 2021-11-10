"""
    Author: Mogage Nicolae
    Creation date: 6 nov 2021
    Modul pentru gestiunea entitatilor
"""
from domain.data import Books, Clients

class BooksService:
    """
        Clasa ce gestioneaza entitatile de tip carte
    """
    def __init__(self, books_repository, books_validator):
        """
            Initializarea unui serviciu de gestiune carti
        :param books_repository: repositoriu de carti
        :param books_validator: validator de carti
        """
        self.__books_repository = books_repository
        self.__books_validator = books_validator

    def add_book(self, id_book, book_title, book_description, book_author):
        """
            Creeaza, valideaza si adauga o carte la lista
        :param id_book: int
        :param book_title: string
        :param book_description: string
        :param book_author: string
        """
        book = Books(id_book, book_title, book_description, book_author)
        self.__books_validator.validate_book(book)
        self.__books_repository.add_book(book)

    def update_book_title(self, id, title):
        """
            Valideaza id-ul si verifica daca exista o carte cu acest id
            Valideaza si actualizeaza titlul
        :param id: int
        :param title: string
        """
        self.__books_validator.validate_id(id)
        self.__books_repository.find_book_by_id(id)
        self.__books_validator.validate_title(title)
        self.__books_repository.update_book_title(id, title)

    def update_book_description(self, id, description):
        """
            Valideaza id-ul si verifica daca exista o carte cu acest id
            Valideaza si actualizeaza descrierea
        :param id: int
        :param description: string
        """
        self.__books_validator.validate_id(id)
        self.__books_repository.find_book_by_id(id)
        self.__books_validator.validate_description(description)
        self.__books_repository.update_book_description(id, description)

    def update_book_author(self, id, author):
        """
            Valideaza id-ul si verifica daca exista o carte cu acest id
            Valideaza si actualizeaza autorul
        :param id: int
        :param author: string
        """
        self.__books_validator.validate_id(id)
        self.__books_repository.find_book_by_id(id)
        self.__books_validator.validate_author(author)
        self.__books_repository.update_book_author(id, author)

    def delete_book_by_id(self, id):
        """
            Valideaza id-ul si sterge cartea cu acest id
        :param id: int
        """
        self.__books_validator.validate_id(id)
        self.__books_repository.delete_book_by_id(id)

    def delete_book_by_title(self, title):
        """
            Valideaza titlul si sterge cartile cu acest titlu
        :param title: string
        """
        self.__books_validator.validate_title(title)
        self.__books_repository.delete_book_by_title(title)

    def delete_book_by_description(self, description):
        """
            Valideaza descrierea si sterge cartile cu aceasta descriere
        :param description: string
        """
        self.__books_validator.validate_description(description)
        self.__books_repository.delete_book_by_description(description)

    def delete_book_by_author(self, author):
        """
            Valideaza autorul si sterge cartile cu acest autor
        :param author: string
        """
        self.__books_validator.validate_author(author)
        self.__books_repository.delete_book_by_author(author)

    def print_all_books(self):
        """
            Afiseaza toate cartile
        """
        self.__books_repository.print_all_books()

    def print_books_id(self, id):
        """
            Valideaza id-ul si afiseaza cartea cu acest id
        :param id: int
        """
        self.__books_validator.validate_id(id)
        self.__books_repository.print_books_id(id)

    def print_books_title(self, title):
        """
            Valideaza titlul si afiseaza cartile cu acest titlu
        :param title: string
        """
        self.__books_validator.validate_title(title)
        self.__books_repository.print_books_title(title)

    def print_books_description(self, description):
        """
            Valideaza descrierea si afiseaza cartile cu aceasta descriere
        :param description: string
        """
        self.__books_validator.validate_description(description)
        self.__books_repository.print_books_description(description)

    def print_books_author(self, author):
        """
            Valideaza autorul si afiseaza cartile cu acest autor
        :param author: string
        """
        self.__books_validator.validate_author(author)
        self.__books_repository.print_books_author(author)

    def check_book_exists(self, id):
        """
            Valideaza id-ul si verifica daca exista o carte cu acest id.
        :param id: int
        """
        self.__books_validator.validate_id(id)
        self.__books_repository.find_book_by_id(id)

    def get_all_books(self):
        """
            Functie de getter a tuturor cartilor
        :return: lista cu toate cartile
        """
        return self.__books_repository.books

class ClientsService:
    """
        Clasa ce se ocupa gestiunea entitatilor de tip client
    """
    def __init__(self, clients_repository, clients_validator):
        """
            Initializarea unui serviciu de gestiune a clientilor
        :param clients_repository: repozitoriu pentru clienti
        :param clients_validator: validator pentru clienti
        """
        self.__clients_repository = clients_repository
        self.__clients_validator = clients_validator

    def add_client(self, id, name, cnp):
        """
            Creeaza, valideaza si adauga un client la lista
        :param id: int
        :param name: string
        :param cnp: int
        """
        client = Clients(id, name, cnp)
        self.__clients_validator.validate_client(client)
        self.__clients_repository.add_client(client)

    def update_client_name(self, id, name):
        """
            Valideaza id-ul si verifica daca exista un client cu acest id
            Valideaza numele si actualizeaza numele cartii
        :param id: int
        :param name: string
        """
        self.__clients_validator.validate_id(id)
        self.__clients_repository.find_client_by_id(id)
        self.__clients_validator.validate_name(name)
        self.__clients_repository.update_name(id, name)

    def delete_client_by_id(self, id):
        """
            Valideaza id-ul si sterge clientul cu acest id
        :param id: int
        """
        self.__clients_validator.validate_id(id)
        self.__clients_repository.delete_by_id(id)

    def delete_client_by_name(self, name):
        """
            Valideaza numele si sterge clientii cu acest nume
        :param name: string
        """
        self.__clients_validator.validate_name(name)
        self.__clients_repository.delete_by_name(name)

    def delete_client_by_cnp(self, cnp):
        """
            Valideaza cnp-ul si sterge clientul cu acest cnp
        :param cnp: int
        """
        self.__clients_validator.validate_cnp(cnp)
        self.__clients_repository.delete_by_cnp(cnp)

    def print_all_clients(self):
        """
            Afiseaza toti clientii
        """
        self.__clients_repository.print_all_clients()

    def print_clients_id(self, id):
        """
            Valideaza id-ul si afiseaza clientul cu acest id
        :param id: int
        """
        self.__clients_validator.validate_id(id)
        self.__clients_repository.print_clients_id(id)

    def print_clients_name(self, name):
        """
            Valideaza numele si afiseaza clientii cu acest nume
        :param name: string
        """
        self.__clients_validator.validate_name(name)
        self.__clients_repository.print_clients_name(name)

    def print_clients_cnp(self, cnp):
        """
            Valideaza cnp-ul si afiseaza clientul cu acest cnp
        :param cnp: int
        """
        self.__clients_validator.validate_cnp(cnp)
        self.__clients_repository.print_clients_cnp(cnp)

    def check_client_exists(self, id):
        """
            Valideaza id-ul si verifica daca exista un client cu acest id
        :param id: int
        """
        self.__clients_validator.validate_id(id)
        self.__clients_repository.find_client_by_id(id)

    def get_all_clients(self):
        """
            Functie de getter a tuturor clientilor
        :return: lista cu toti clientii
        """
        return self.__clients_repository.clients
