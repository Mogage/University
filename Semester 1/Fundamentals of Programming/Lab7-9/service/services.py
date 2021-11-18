"""
    Author: Mogage Nicolae
    Creation date: 6 nov 2021
    Modul pentru gestiunea entitatilor
"""
import datetime

from domain.data import Books, Clients, Rent

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
        self.__books_repository.rewrite_file()

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
        self.__books_repository.rewrite_file()

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
        self.__books_repository.rewrite_file()

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
        self.__books_repository.rewrite_file()

    def delete_book_by_id(self, id):
        """
            Valideaza id-ul si sterge cartea cu acest id
        :param id: int
        """
        self.__books_validator.validate_id(id)
        self.__books_repository.delete_book_by_id(id)
        self.__books_repository.rewrite_file()

    def delete_book_by_title(self, title):
        """
            Valideaza titlul si sterge cartile cu acest titlu
        :param title: string
        """
        self.__books_validator.validate_title(title)
        self.__books_repository.delete_book_by_title(title)
        self.__books_repository.rewrite_file()

    def delete_book_by_description(self, description):
        """
            Valideaza descrierea si sterge cartile cu aceasta descriere
        :param description: string
        """
        self.__books_validator.validate_description(description)
        self.__books_repository.delete_book_by_description(description)
        self.__books_repository.rewrite_file()

    def delete_book_by_author(self, author):
        """
            Valideaza autorul si sterge cartile cu acest autor
        :param author: string
        """
        self.__books_validator.validate_author(author)
        self.__books_repository.delete_book_by_author(author)
        self.__books_repository.rewrite_file()

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

    def find_book_by_id(self, id):
        """
            Valideaza id-ul si verifica daca exista o carte cu acest id.
        :param id: int
        """
        self.__books_validator.validate_id(id)
        return self.__books_repository.find_book_by_id(id)

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
        self.__clients_repository.rewrite_file()

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
        self.__clients_repository.rewrite_file()

    def delete_client_by_id(self, id):
        """
            Valideaza id-ul si sterge clientul cu acest id
        :param id: int
        """
        self.__clients_validator.validate_id(id)
        self.__clients_repository.delete_by_id(id)
        self.__clients_repository.rewrite_file()

    def delete_client_by_name(self, name):
        """
            Valideaza numele si sterge clientii cu acest nume
        :param name: string
        """
        self.__clients_validator.validate_name(name)
        self.__clients_repository.delete_by_name(name)
        self.__clients_repository.rewrite_file()

    def delete_client_by_cnp(self, cnp):
        """
            Valideaza cnp-ul si sterge clientul cu acest cnp
        :param cnp: int
        """
        self.__clients_validator.validate_cnp(cnp)
        self.__clients_repository.delete_by_cnp(cnp)
        self.__clients_repository.rewrite_file()

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

    def print_filter_name_by_prefix(self, prefix):
        """
            Valideaza prefixul si filtreaza lista de clienti, afisand clientii a caror nume incepe cu acest prefix
        :param prefix: string
        """
        self.__clients_validator.validate_prefix(prefix)
        self.__clients_repository.print_filter_name_by_prefix(prefix)

    def find_client_by_id(self, id):
        """
            Valideaza id-ul si verifica daca exista un client cu acest id
        :param id: int
        """
        self.__clients_validator.validate_id(id)
        return self.__clients_repository.find_client_by_id(id)

    def get_all_clients(self):
        """
            Functie de getter a tuturor clientilor
        :return: lista cu toti clientii
        """
        return self.__clients_repository.clients

class RentService:
    """
        Clasa pentru gestiunea entitatilor inchiriere
    """
    def __init__(self, rent_repository, rent_validator, books_service, clients_service):
        """
            Initializarea datelor pentru gestiunea inchirierilor
        :param rent_repository: repo inchirieri
        :param rent_validator: validator inchirieri
        :param books_service: serviciu carti
        :param clients_service: serviciu clienti
        """
        self.__books_service = books_service
        self.__clients_service = clients_service
        self.__rent_repository = rent_repository
        self.__rent_validator = rent_validator

    def check_book_exists(self, id):
        """
            Verifica daca o carte exista
        :param id: int
        """
        self.__books_service.find_book_by_id(id)

    def check_client_exists(self, id):
        """
            Verifica daca un client exista
        :param id: int
        """
        self.__clients_service.find_client_by_id(id)

    def check_rent_client_book_exists(self, id_book, id_client):
        """
            Verifica daca clientul cu id-ul id_client a inchiriat cartea cu id-ul id_book
        :param id_book: int
        :param id_client: int
        """
        self.check_book_exists(id_book)
        self.check_client_exists(id_client)
        self.__rent_repository.check_rent_by_client_id(id_client)
        self.__rent_repository.check_rent_client_book(id_book, id_client)

    def check_rent_client_exists(self, id):
        """
            Verifica daca clientul cu id-ul id a inchiriat cel putin o carte
        :param id: int
        """
        self.check_client_exists(id)
        self.__rent_repository.check_rent_by_client_id(id)

    def add_rent(self, id_book, id_client):
        """
            Cauta cartea si clientul cu aceste id-uri si adauga in repo inchirierea
            creeata la data din momentul inchirierii
        :param id_book: int
        :param id_client: int
        """
        date = datetime.datetime.now().strftime("%d/%m/%y")
        book = self.__books_service.find_book_by_id(id_book)
        client = self.__clients_service.find_client_by_id(id_client)
        rent = Rent(book, client, date)
        self.__rent_repository.add_rent(rent)
        self.__rent_repository.rewrite_file()

    def delete_rent(self, id_book, id_client):
        """
            Daca clinentul cu id-ul id_client a inchiriat cartea cu id-ul id_book sterge aceasta inchiriere din repo
        :param id_book: int
        :param id_client: int
        """
        self.check_rent_client_book_exists(id_book, id_client)
        self.__rent_repository.delete_rent(id_book, id_client)
        self.__rent_repository.rewrite_file()

    def print_all(self):
        """
            Afiseaza toate inchirierile efectuate
        """
        self.__rent_repository.print_all()

    def print_rent_by_client_id(self, id):
        """
            Afiseaza toate inchirierile efectuate de clientul cu id-ul id
        :param id: int
        """
        self.__clients_service.find_client_by_id(id)
        self.__rent_repository.print_rent_by_client_id(id)

    def get_all_rent(self):
        """
            Getter pentru toate inchirierile
        :return: list []
        """
        return self.__rent_repository.rent
