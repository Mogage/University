"""
    Author: Mogage Nicolae
    Creation date: 22 nov 2021
    Modul pentru gestiunea entitatilor carte
"""
import random
import string

from domain.data import Books

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

    @staticmethod
    def __generate_data():
        """
            Genereaza datele pentru entitati carte
        :return: int, string, string, string
        """
        id = random.randint(0, 1000)
        title = ''.join(random.choice(string.ascii_letters) for _ in range(id % 50))
        description = ''.join(random.choice(string.ascii_letters) for _ in range(id % 100))
        author = ''.join(random.choice(string.ascii_letters) for _ in range(id % 50))
        return id, title, description, author

    def generate_books(self, number_of_books):
        """
            Genereaza number_of_books entitati de tip client
        :param number_of_books: int - numar de carti care sa se genereze
        """
        index = 0
        while index < number_of_books:
            id, title, description, author = self.__generate_data()
            book = Books(id, title, description, author)
            self.__books_validator.validate_book(book)
            self.__books_repository.add_book(book)
            print("Carte adaugat cu succes.")
            index = index + 1

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
