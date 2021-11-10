"""
    Author: Mogage Nicolae
    Creation date: 6 nov 2021
    Modul pentru stocarea entitatilor
"""
from error.errors import BookRepositoryError, ClientRepositoryError

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

class ClientsRepository:
    """
        Clasa ce se ocupa cu stocarea clientilor
    """
    def __init__(self):
        """
            Functie de initializare a unei liste entitati de tip client
        """
        self.__clients = []

    @property
    def clients(self):
        """
            Functie de getter pentru lista de entitati
        :return: __clients
        """
        return self.__clients

    def find_client_by_id(self, id):
        """
            Cauta un client cu id-ul id
        :param id: id > 0
        :return: entitatea de tip client cu id-ul id
        :raise: Id client inexistent. daca nu exista un client cu acest id
        """
        for _client in self.__clients:
            if _client.id == id:
                return _client
        raise ClientRepositoryError("Id client inexistent.")

    def find_client_by_cnp(self, cnp):
        """
            Cauta un client cu cnp-ul cnp
        :param cnp: int cu exact 13 cifre
        :return: entitate de tip client cu cnp-ul cnp
        :raise: Cnp client inexistent. daca nu exista un client cu acest cnp.
        """
        for _client in self.__clients:
            if _client.cnp == cnp:
                return _client
        raise ClientRepositoryError("Cnp client inexistent.")

    def add_client(self, client):
        """
            Adauga un client inexistent in repertoriu, un client exista deja daca este alt client cu acelasi id sau cnp
        :param client: un client
        :return: -
        :raises: Exista deja un client cu acest id. daca s-a gasit un client cu acelasi id
                 Exista deja un client cu acest cnp. daca s-a gasit un client cu acelasi cnp
        """
        for _client in self.__clients:
            if _client.id == client.id:
                raise ClientRepositoryError("Exista deja un client cu acest id.")
            
        for _client in self.__clients:
            if _client.cnp == client.cnp:
                raise ClientRepositoryError("Exista deja un client cu acest cnp.")

        index = 0
        number_of_clients = len(self.__clients)
        while index < number_of_clients and client > self.__clients[index]:
            index = index + 1

        self.__clients.insert(index, client)

    def update_name(self, id, name):
        """
            Actualizeaza numele unui client cu id-ul id
        :param id: int > 0
        :param name: not empty string
        :return: -
        :raise: Id client inexistent. daca nu exista un client cu acest id.
        """
        index = 0
        number_of_clients = len(self.__clients)
        while index < number_of_clients:
            if self.__clients[index].id == id:
                self.__clients[index].name = name
                return
            index = index + 1
        raise ClientRepositoryError("Id client inexistent.")

    def delete_by_id(self, id):
        """
            Sterge un client cu id-ul id
        :param id: int > 0
        :return: -
        :raise: Id client inexistent. daca nu exista un client cu acest id.
        """
        index = 0
        number_of_clients = len(self.__clients)
        while index < number_of_clients:
            if self.__clients[index].id == id:
                self.__clients.pop(index)
                return
            index = index + 1

        raise ClientRepositoryError("Id client inexistent.")

    def delete_by_name(self, name):
        """
            Sterge clientii cu numele name
        :param name: not empty string
        :return: -
        :raise: Nume client inexistent. daca nu exista clienti cu acest nume
        """
        index = 0
        number_of_clients = len(self.__clients)
        client_is_deleted = False
        while index < number_of_clients:
            if self.__clients[index].name == name:
                self.__clients.pop(index)
                index = index - 1
                number_of_clients = number_of_clients - 1
                client_is_deleted = True
            index = index + 1

        if not client_is_deleted:
            raise ClientRepositoryError("Nume client inexistent.")

    def delete_by_cnp(self, cnp):
        """
            Sterge clientul cu cnp-ul cnp
        :param cnp: int cu exact 13 cifre
        :return: -
        :raise: Cnp client inexistent. daca nu exista un client cu acest cnp
        """
        index = 0
        number_of_clients = len(self.__clients)
        while index < number_of_clients:
            if self.__clients[index].cnp == cnp:
                self.__clients.pop(index)
                return
            index = index + 1

        raise ClientRepositoryError("Cnp client inexistent.")

    def print_all_clients(self):
        """
            Afiseaza toti clientii
        :return: -
        """
        for _client in self.__clients:
            print(str(_client))

    def print_clients_id(self, id):
        """
            Afiseaza clienul cu id-ul id
        :param id: int > 0
        :return: -
        :raise: Id carte inexistent. daca nu exista un client cu acest id
        """
        for _client in self.__clients:
            if _client.id == id:
                print(str(_client))
                return
        raise ClientRepositoryError("Id carte inexistent.")

    def print_clients_name(self, name):
        """
            Afiseaza clientii cu numele name
        :param name: not empty string
        :return: -
        :raise: Nume client inexistent. daca nu exista clienti cu acest nume
        """
        client_exists = False
        for _client in self.__clients:
            if _client.name == name:
                print(str(_client))
                client_exists = True

        if not client_exists:
            raise ClientRepositoryError("Nume client inexistent.")

    def print_clients_cnp(self, cnp):
        """
            Afiseaza clientul cu cnp-ul cnp
        :param cnp: not empty string
        :return: -
        :raise: Cnp client inexistent. daca nu exista un client cu acest cnp
        """
        for _client in self.__clients:
            if _client.cnp == cnp:
                print(str(_client))
                return
        raise ClientRepositoryError("Cnp client inexistent.")

    def __len__(self):
        """
            Lungimea unui repozitoriu este egala cu numarul de clienti inregistrati
        :return: int - numarul de clienti inregistrati
        """
        return len(self.__clients)
