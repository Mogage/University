"""
    Author: Mogage Nicolae
    Creation date: 6 nov 2021
    Modul pentru stocarea entitatilor carte
"""
from domain.data import Books
from error.errors import BookRepositoryError

class BooksRepository:
    """
        Clasa ce se ocupa cu stocarea cartilor
    """
    def __init__(self):
        """
            Functia de initalizare a unei liste de entitati de tip carte
        """
        self.__books = []

    @property
    def books(self):
        """
            Functie de getter pentru carti
        :return: __books
        """
        return self.__books

    def add_book(self, book):
        """
            Functie de adaugare in repozitoriu a unei carti, lista sortata dupa id
        :param book: o carte
        :return: -
        :raise: Exista deja o carte cu acest id. daca exista deja o carte cu acelasi id
        """
        if book in self.__books:
            raise BookRepositoryError("Exista deja o carte cu acest id.")

        index = 0
        number_of_books = len(self.__books)
        while index < number_of_books and book > self.__books[index]:
            index = index + 1

        self.__books.insert(index, book)

    def update_book_title(self, id, title):
        """
            Functie de actualizare a titlului unei carti cu id-ul id
        :param id: int > 0
        :param title: not empty string
        :return: -
        :raise: Id carte inexistent. daca nu exista o carte cu id-ul id
        """
        index = 0
        number_of_books = len(self.__books)
        while index < number_of_books:
            if self.__books[index].id == id:
                self.__books[index].title = title
                return
            index = index + 1
        raise BookRepositoryError("Id carte inexistent.")

    def update_book_description(self, id, description):
        """
            Functie de actualizare a descrierii unei carti cu id-ul id
        :param id: int > 0
        :param description: not empty string
        :return: -
        :raise: Id carte inexistent. daca nu exista o carte cu id-ul id
        """
        index = 0
        number_of_books = len(self.__books)
        while index < number_of_books:
            if self.__books[index].id == id:
                self.__books[index].description = description
                return
            index = index + 1
        raise BookRepositoryError("Id carte inexistent.")

    def update_book_author(self, id, author):
        """
            Functie de actualizare a autorului unei carti cu id-ul id
        :param id: id > 0
        :param author: not empty string
        :return: -
        :raise: Id carte inexistent. daca nu exista o carte cu id-ul id
        """
        index = 0
        number_of_books = len(self.__books)
        while index < number_of_books:
            if self.__books[index].id == id:
                self.__books[index].author = author
                return
            index = index + 1
        raise BookRepositoryError("Id carte inexistent.")

    def delete_book_by_id(self, id):
        """
            Functie de stergere a unei carti cu id-ul id
        :param id: int > 0
        :return: -
        :raise: Id carte inexistent. daca nu exista o carte cu id-ul id
        """
        index = 0
        number_of_books = len(self.__books)
        while index < number_of_books:
            if self.__books[index].id == id:
                self.__books.pop(index)
                return
            index = index + 1

        raise BookRepositoryError("Id carte inexistent.")

    def delete_book_by_title(self, title):
        """
            Functie de stergere a cartilor cu titlul title
        :param title: not empty string
        :return: -
        :raise: Titlu carte inexistent. daca nu exista nicio carte cu acest titlu
        """
        index = 0
        number_of_books = len(self.__books)
        book_is_deleted = False
        while index < number_of_books:
            if self.__books[index].title == title:
                self.__books.pop(index)
                book_is_deleted = True
                index = index - 1
                number_of_books = number_of_books - 1
            index = index + 1

        if not book_is_deleted:
            raise BookRepositoryError("Titlu carte inexistent.")

    def delete_book_by_description(self, description):
        """
            Functie de stergere a cartilor cu descrierea description
        :param description: not empty string
        :return: -
        :raise: Descriere carte inexistenta. daca nu exista nicio carte cu aceasta descriere
        """
        index = 0
        number_of_books = len(self.__books)
        book_is_deleted = False
        while index < number_of_books:
            if self.__books[index].description == description:
                self.__books.pop(index)
                book_is_deleted = True
                index = index - 1
                number_of_books = number_of_books - 1
            index = index + 1

        if not book_is_deleted:
            raise BookRepositoryError("Descriere carte inexistenta.")

    def delete_book_by_author(self, author):
        """
            Functie de stergere a cartilor cu autorul author
        :param author: not empty string
        :return: -
        :raise: Autor carte inexistent. daca nu exista nicio carte cu acest autor
        """
        index = 0
        number_of_books = len(self.__books)
        book_is_deleted = False
        while index < number_of_books:
            if self.__books[index].author == author:
                self.__books.pop(index)
                book_is_deleted = True
                index = index - 1
                number_of_books = number_of_books - 1
            index = index + 1

        if not book_is_deleted:
            raise BookRepositoryError("Autor carte inexistent.")

    def print_all_books(self):
        """
            Functie de afisare a tuturor cartilor
        :return: -
        """
        for _book in self.__books:
            print(str(_book))

    def print_books_id(self, id):
        """
            Functie de afisare a cartii cu id-ul id
        :param id: int > 0
        :return: -
        :raise: Id carte inexistent. daca nu exista o carte cu acest id
        """
        for _book in self.__books:
            if _book.id == id:
                print(str(_book))
                return
        raise BookRepositoryError("Id carte inexistent.")

    def print_books_title(self, title):
        """
            Functie de afisare a cartii cu titlul title
        :param title: not empty string
        :return: -
        :raise: Titlu carte inexistent. daca nu exista o carte cu acest titlu
        """
        book_exists = False
        for _book in self.__books:
            if _book.title == title:
                print(str(_book))
                book_exists = True

        if not book_exists:
            raise BookRepositoryError("Titlu carte inexistent.")

    def print_books_description(self, description):
        """
            Functie de afisare a cartii cu descrierea description
        :param description: not empty string
        :return: -
        :raise: Descriere carte inexistenta. daca nu exista o carte cu aceasta descriere
        """
        book_exists = False
        for _book in self.__books:
            if _book.description == description:
                print(str(_book))
                book_exists = True

        if not book_exists:
            raise BookRepositoryError("Descriere carte inexistenta.")

    def print_books_author(self, author):
        """
            Functie de afisare a cartii cu autorul author
        :param author: not empty string
        :return: -
        :raise: Autor carte inexistent. daca nu exista o carte cu acest autor
        """
        book_exists = False
        for _book in self.__books:
            if _book.author == author:
                print(str(_book))
                book_exists = True

        if not book_exists:
            raise BookRepositoryError("Autor carte inexistent.")

    def find_book_by_id(self, id):
        """
            Cauta o carte cu id-ul id si o returneaza daca exista
        :param id: int > 0
        :return: entitate de tip carte cu id-ul id
        :raise: Id carte inexistent. daca nu exista o carte cu acest id
        """
        for _book in self.__books:
            if _book.id == id:
                return _book
        raise BookRepositoryError("Id carte inexistent.")

    def __len__(self):
        """
            Lungimea unui repozitoriu este egala cu numarul de carti inregistrate
        :return: int - numarul de carti inregistrate
        """
        return len(self.__books)

class BooksFileRepository(BooksRepository):
    """
        Clasa pentru stocarea in fisiere a cartilor
    """
    def __init__(self, file_name):
        """
            Initializarea datelor pentru stocarea in fisiere
        :param file_name: nume fisier - string
        """
        self.__file_name = file_name
        BooksRepository.__init__(self)
        self.__load_from_file()

    @staticmethod
    def __create_book_from_file(line):
        """
            Creeaza o carte din string-ul line in care atributele sunt despartite prin ;
        :param line: o linie formatata din fisier - string
        :return: o carte
        """
        fields = line.split(";")
        fields[3] = fields[3].removesuffix("\n")
        book = Books(int(fields[0]), fields[1], fields[2], fields[3])
        return book

    def __load_from_file(self):
        """
            Incarca toate datele din fisier
        """
        with open(self.__file_name) as book_file:
            for line in book_file:
                book = self.__create_book_from_file(line)
                BooksRepository.add_book(self, book)

    @staticmethod
    def __create_line(book):
        """
            Formateaza o carte pentru a fi stocata in fisier
        :param book: o carte
        :return: un string cu atributele separate prin ;
        """
        return str(book.id) + ";" + book.title + ";" + book.description + ";" + book.author

    @staticmethod
    def __add_to_file(line, book_file):
        """
            Scrie in fisier cartea formatata pentru o linie
        :param line: carte formatata sub forma de linie - string
        :param book_file: fisierul in care se vor stoca
        """
        book_file.write(line)
        book_file.write("\n")

    def rewrite_file(self):
        """
            Scrie in fisier toate cartile
        """
        with open(self.__file_name, "w") as book_file:
            for _book in self.books:
                line = self.__create_line(_book)
                self.__add_to_file(line, book_file)
