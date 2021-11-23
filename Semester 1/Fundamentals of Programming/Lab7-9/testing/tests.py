import datetime

from domain.data import Books, Clients
from domain.dtos import RentDto
from error.errors import BookValidationError, BookRepositoryError, ClientValidationError, ClientRepositoryError, \
    RentRepositoryError, RentValidationError
from repository.book_repository import BooksFileRepository, BooksRepository
from repository.client_repository import ClientsRepository, ClientsFileRepository
from repository.rent_repository import RentFileRepository
from service.book_service import BooksService
from service.client_service import ClientsService
from service.rent_service import RentService
from validator.validators import BooksValidator, ClientsValidator, RentValidator

class Tests:

    @staticmethod
    def __test_create_book(id_book, book_title, book_description, book_author):
        book = Books(id_book, book_title, book_description, book_author)
        assert book.id == id_book
        assert book.title == book_title
        assert book.description == book_description
        assert book.author == book_author
        book.description = "Assembly"
        assert book.description == "Assembly"
        return book

    @staticmethod
    def __test_equal_books(book, id_book, other_title, book_description, book_author):
        other_book = Books(id_book, other_title, book_description, book_author)
        assert other_book == book

    @staticmethod
    def __test_print_book(book):
        assert str(book) == "[1]Titlu: Arhitectura calculatoarelor\n" \
                            "Descriere: Assembly\n" \
                            "Autor: Alexandru Vancea"

    def __run_tests_create_book(self):
        id_book = 1
        book_title = "Arhitectura calculatoarelor"
        book_description = "ASC stuff"
        book_author = "Alexandru Vancea"

        other_title = "Fundamentele programarii"

        book = self.__test_create_book(id_book, book_title, book_description, book_author)
        self.__test_equal_books(book, id_book, other_title, book_description, book_author)
        self.__test_print_book(book)

    @staticmethod
    def __test_validate_book_success(book_validator, id_book, book_title, book_description, book_author):
        book = Books(id_book, book_title, book_description, book_author)
        try:
            book_validator.validate_book(book)
        except BookValidationError:
            assert False

    @staticmethod
    def __test_validate_book_invalid(book_validator, invalid_id_book, invalid_book_title, book_description,
                                     book_author):
        book = Books(invalid_id_book, invalid_book_title, book_description, book_author)
        try:
            book_validator.validate_book(book)
            assert False
        except BookValidationError as bve:
            assert str(bve) == "Id-ul introdus este invalid.\nTitlul introdus este invalid.\n"

    @staticmethod
    def __test_validate_book_specifications_success(book_validator, id_book, book_author):
        try:
            book_validator.validate_id(id_book)
        except BookValidationError:
            assert False

        try:
            book_validator.validate_author(book_author)
        except BookValidationError:
            assert False

    @staticmethod
    def __test_validate_book_specifications_invalid(bool_validator, invalid_id_book, invalid_book_title):
        try:
            bool_validator.validate_id(invalid_id_book)
            assert False
        except BookValidationError as bve:
            assert str(bve) == "Id-ul introdus este invalid."

        try:
            bool_validator.validate_title(invalid_book_title)
            assert False
        except BookValidationError as bve:
            assert str(bve) == "Titlul introdus este invalid."

    def __run_tests_validate_book(self):
        book_validator = BooksValidator()
        id_book = 1
        book_title = "Arhitectura calculatoarelor"
        book_description = "ASC stuff"
        book_author = "Alexandru Vancea"

        invalid_id_book = -1
        invalid_book_title = ""

        self.__test_validate_book_success(book_validator, id_book, book_title, book_description, book_author)
        self.__test_validate_book_invalid(book_validator, invalid_id_book, invalid_book_title, book_description,
                                          book_author)
        self.__test_validate_book_specifications_success(book_validator, id_book, book_author)
        self.__test_validate_book_specifications_invalid(book_validator, invalid_id_book, invalid_book_title)

    @staticmethod
    def __test_repository_add_book(book_repository, book):
        book_repository.add_book(book)
        assert len(book_repository.books) == 1
        assert book_repository.books[0].id == book.id
        assert book_repository.books[0].title == book.title
        assert book_repository.books[0].author == book.author

    @staticmethod
    def __test_repository_add_existent_id_book(book_repository, other_book):
        try:
            book_repository.add_book(other_book)
            assert False
        except BookRepositoryError as bre:
            assert str(bre) == "Exista deja o carte cu acest id."

    @staticmethod
    def __test_find_book_by_existent_id(book_repository, book, id_book):
        found_book = book_repository.find_book_by_id(id_book)
        assert book.id == found_book.id
        assert book.title == found_book.title
        assert book.description == found_book.description
        assert book.author == found_book.author

    @staticmethod
    def __test_find_book_by_non_existent_id(book_repository, non_existent_id_book):
        try:
            book_repository.find_book_by_id(non_existent_id_book)
            assert False
        except BookRepositoryError as bre:
            assert str(bre) == "Id carte inexistent."

    @staticmethod
    def __test_repository_delete_book_by_existent_data(book_repository, book, id_book, book_title,
                                                       book_description, book_author):
        book_repository.delete_book_by_id(id_book)
        assert len(book_repository) == 0
        book_repository.add_book(book)

        book_repository.delete_book_by_title(book_title)
        assert len(book_repository) == 0
        book_repository.add_book(book)

        book_repository.delete_book_by_description(book_description)
        assert len(book_repository) == 0
        book_repository.add_book(book)

        book_repository.delete_book_by_author(book_author)
        assert len(book_repository) == 0
        book_repository.add_book(book)

    @staticmethod
    def __test_repository_delete_book_by_non_existent_data(book_repository, non_existent_id_book,
                                                           non_existent_book_title):
        try:
            book_repository.delete_book_by_id(non_existent_id_book)
            assert False
        except BookRepositoryError as bre:
            assert str(bre) == "Id carte inexistent."

        try:
            book_repository.delete_book_by_title(non_existent_book_title)
            assert False
        except BookRepositoryError as bre:
            assert str(bre) == "Titlu carte inexistent."

    @staticmethod
    def __test_repository_update_book_success(book_repository, id_book, other_title, other_description):
        book_repository.update_book_title(id_book, other_title)
        book_repository.update_book_description(id_book, other_description)
        assert book_repository.books[0].id == id_book
        assert book_repository.books[0].title == other_title
        assert book_repository.books[0].description == other_description

    @staticmethod
    def __test_repository_update_book_non_existent_id(book_repository, non_existent_id_book, book_title):
        try:
            book_repository.update_book_title(non_existent_id_book, book_title)
            assert False
        except BookRepositoryError as bre:
            assert str(bre) == "Id carte inexistent."

    def __run_tests_repository_book(self):
        book_repository = BooksRepository()
        id_book = 1
        non_existent_id_book = 5
        non_existent_book_title = "asc"
        book_title = "Arhitectura calculatoarelor"
        book_description = "ASC stuff"
        book_author = "Alexandru Vancea"
        book = Books(id_book, book_title, book_description, book_author)

        other_title = "Fundamentele programarii"
        other_description = "Python stuff"
        other_book = Books(id_book, other_title, other_description, book_author)

        self.__test_repository_add_book(book_repository, book)
        self.__test_repository_add_existent_id_book(book_repository, other_book)
        self.__test_find_book_by_existent_id(book_repository, book, id_book)
        self.__test_find_book_by_non_existent_id(book_repository, non_existent_id_book)
        self.__test_repository_update_book_success(book_repository, id_book, other_title, other_description)
        self.__test_repository_update_book_non_existent_id(book_repository, non_existent_id_book, book_title)
        book = Books(id_book, book_title, book_description, book_author)
        self.__test_repository_delete_book_by_existent_data(book_repository, book, id_book, book_title,
                                                            book_description, book_author)
        self.__test_repository_delete_book_by_non_existent_data(book_repository, non_existent_id_book,
                                                                non_existent_book_title)

    @staticmethod
    def __test_service_add_book_success(book_service, id_book, book_title, book_description, book_author):
        assert len(book_service.get_all_books()) == 0

        book_service.add_book(id_book, book_title, book_description, book_author)
        books = book_service.get_all_books()
        assert len(books) == 1
        assert books[0].id == id_book
        assert books[0].author == book_author

    @staticmethod
    def __test_service_add_book_invalid(book_service, invalid_id_book, invalid_book_title,
                                        invalid_book_description, invalid_book_author):
        try:
            book_service.add_book(invalid_id_book, invalid_book_title, invalid_book_description, invalid_book_author)
            assert False
        except BookValidationError as bve:
            assert str(bve) == "Id-ul introdus este invalid.\nTitlul introdus este invalid.\n" \
                               "Descrierea introdusa este invalida.\nAutorul introdus este invalid.\n"

    @staticmethod
    def __test_service_delete_book_success(book_service, id_book, book_title, book_description, book_author):
        book_service.delete_book_by_id(id_book)
        assert len(book_service.get_all_books()) == 0
        book_service.add_book(id_book, book_title, book_description, book_author)

        book_service.delete_book_by_author(book_author)
        assert len(book_service.get_all_books()) == 0
        book_service.add_book(id_book, book_title, book_description, book_author)

    @staticmethod
    def __test_service_delete_book_invalid(book_service, invalid_id_book, invalid_book_title):
        try:
            book_service.delete_book_by_id(invalid_id_book)
            assert False
        except BookValidationError as bve:
            assert str(bve) == "Id-ul introdus este invalid."

        try:
            book_service.delete_book_by_title(invalid_book_title)
            assert False
        except BookValidationError as bve:
            assert str(bve) == "Titlul introdus este invalid."

    @staticmethod
    def __test_service_delete_book_non_existent(book_service, non_existent_id, non_existent_description,
                                                non_existent_author):
        try:
            book_service.delete_book_by_id(non_existent_id)
            assert False
        except BookRepositoryError as bre:
            assert str(bre) == "Id carte inexistent."

        try:
            book_service.delete_book_by_description(non_existent_description)
            assert False
        except BookRepositoryError as bre:
            assert str(bre) == "Descriere carte inexistenta."

        try:
            book_service.delete_book_by_author(non_existent_author)
            assert False
        except BookRepositoryError as bre:
            assert str(bre) == "Autor carte inexistent."

    @staticmethod
    def __test_service_update_book_succes(book_service, id_book, other_title):
        book_service.update_book_title(id_book, other_title)
        books = book_service.get_all_books()
        assert books[0].title == other_title

    @staticmethod
    def __test_service_update_book_invalid(book_service, id_book, invalid_id_book, invalid_book_title):
        try:
            book_service.update_book_title(invalid_id_book, invalid_book_title)
            assert False
        except BookValidationError as bve:
            assert str(bve) == "Id-ul introdus este invalid."

        try:
            book_service.update_book_title(id_book, invalid_book_title)
            assert False
        except BookValidationError as bve:
            assert str(bve) == "Titlul introdus este invalid."

    @staticmethod
    def __test_service_update_book_non_existent(book_service, non_existent_id_book, non_existent_author):
        try:
            book_service.update_book_author(non_existent_id_book, non_existent_author)
            assert False
        except BookRepositoryError as bre:
            assert str(bre) == "Id carte inexistent."

    def __run_tests_service_book(self):
        book_repository = BooksFileRepository("Save_files/test.txt")
        book_validator = BooksValidator()
        book_service = BooksService(book_repository, book_validator)
        id_book = 1
        book_title = "Arhitectura calculatoarelor"
        book_description = "ASC stuff"
        book_author = "Alexandru Vancea"

        other_title = "Fundamentele programarii"

        invalid_id_book = -1
        invalid_book_title = ""
        invalid_book_description = ""
        invalid_book_author = ""

        non_existent_id_book = 2
        non_existent_description = "Python"
        non_existent_author = "Gabi Mircea"

        self.__test_service_add_book_success(book_service, id_book, book_title, book_description, book_author)
        self.__test_service_add_book_invalid(book_service, invalid_id_book, invalid_book_title,
                                             invalid_book_description, invalid_book_author)
        self.__test_service_delete_book_success(book_service, id_book, book_title, book_description, book_author)
        self.__test_service_delete_book_invalid(book_service, invalid_id_book, invalid_book_title)
        self.__test_service_delete_book_non_existent(book_service, non_existent_id_book, non_existent_description,
                                                     non_existent_author)
        self.__test_service_update_book_succes(book_service, id_book, other_title)
        self.__test_service_update_book_invalid(book_service, id_book, invalid_id_book, invalid_book_title)
        self.__test_service_update_book_non_existent(book_service, non_existent_id_book, non_existent_author)

    @staticmethod
    def __test_create_client(client_id, client_name, client_cnp):
        client = Clients(client_id, client_name, client_cnp)
        assert client.id == client_id
        assert client.name == client_name
        assert client.cnp == client_cnp
        client.name = "Andrei"
        assert client.name == "Andrei"
        return client

    @staticmethod
    def __test_equal_clients(client, client_id, other_name, client_cnp):
        other_client = Clients(client_id, other_name, client_cnp)
        assert client == other_client

    @staticmethod
    def __test_print_client(client):
        assert str(client) == "[1]Nume: Andrei. CNP: 5112233330000"

    def __run_tests_create_client(self):
        client_id = 1
        client_name = "Gabi"
        client_cnp = 5112233330000

        other_name = "Alexandru"

        client = self.__test_create_client(client_id, client_name, client_cnp)
        self.__test_equal_clients(client, client_id, other_name, client_cnp)
        self.__test_print_client(client)

    @staticmethod
    def __test_validate_client_success(client_validator, client_id, client_name, client_cnp):
        client = Clients(client_id, client_name, client_cnp)
        try:
            client_validator.validate_client(client)
        except ClientValidationError:
            assert False

    @staticmethod
    def __test_validate_client_invalid(client_validator, invalid_client_id, invalid_client_name,
                                       invalid_client_cnp):
        client = Clients(invalid_client_id, invalid_client_name, invalid_client_cnp)
        try:
            client_validator.validate_client(client)
            assert False
        except ClientValidationError as cve:
            assert str(
                cve) == "Id-ul introdus este invalid.\nNumele introdus este invalid.\nCnp-ul introdus este invalid.\n"

    @staticmethod
    def __test_validate_client_specifications_succes(client_validator, client_id, client_cnp):
        try:
            client_validator.validate_id(client_id)
        except ClientValidationError:
            assert False

        try:
            client_validator.validate_cnp(client_cnp)
        except ClientValidationError:
            assert False

    @staticmethod
    def __test_validate_client_specifications_invalid(client_validator, invalid_client_name,
                                                      invalid_client_cnp_2):
        try:
            client_validator.validate_name(invalid_client_name)
            assert False
        except ClientValidationError as cve:
            assert str(cve) == "Numele introdus este invalid."

        try:
            client_validator.validate_cnp(invalid_client_cnp_2)
            assert False
        except ClientValidationError as cve:
            assert str(cve) == "Cnp-ul introdus este invalid."

    def __run_tests_validate_client(self):
        client_validator = ClientsValidator()
        client_id = 1
        client_name = "Gabi"
        client_cnp = 5112233330000

        invalid_client_id = -1
        invalid_client_name = ""
        invalid_client_cnp = 25
        invalid_client_cnp_2 = -5112233330000

        self.__test_validate_client_success(client_validator, client_id, client_name, client_cnp)
        self.__test_validate_client_invalid(client_validator, invalid_client_id, invalid_client_name,
                                            invalid_client_cnp)
        self.__test_validate_client_specifications_succes(client_validator, client_id, client_cnp)
        self.__test_validate_client_specifications_invalid(client_validator, invalid_client_name, invalid_client_cnp_2)

    @staticmethod
    def __test_repository_add_client(client_repository, client):
        client_repository.add_client(client)
        assert len(client_repository) == 1
        assert client_repository.clients[0].id == client.id
        assert client_repository.clients[0].name == client.name
        assert client_repository.clients[0].cnp == client.cnp

    @staticmethod
    def __test_repository_add_existing_client(client_repository, other_client):
        try:
            client_repository.add_client(other_client)
            assert False
        except ClientRepositoryError as cre:
            assert str(cre) == "Exista deja un client cu acest id."

    @staticmethod
    def __test_repository_update_client_succes(client_repository, client_id, other_name):
        client_repository.update_name(client_id, other_name)
        assert client_repository.clients[0].name == other_name

    @staticmethod
    def __test_repository_update_client_invalid(client_repository, non_existent_id, other_name):
        try:
            client_repository.update_name(non_existent_id, other_name)
            assert False
        except ClientRepositoryError as cre:
            assert str(cre) == "Id client inexistent."

    @staticmethod
    def __test_repository_delete_client_by_existent_data(client_repository, client, client_id, client_name,
                                                         client_cnp):
        client_repository.delete_by_id(client_id)
        assert len(client_repository) == 0
        client_repository.add_client(client)

        client_repository.delete_by_name(client_name)
        assert len(client_repository) == 0
        client_repository.add_client(client)

        client_repository.delete_by_cnp(client_cnp)
        assert len(client_repository) == 0
        client_repository.add_client(client)

    @staticmethod
    def __test_repository_delete_client_by_non_existent_data(client_repository, non_existent_id, other_name):
        try:
            client_repository.delete_by_id(non_existent_id)
            assert False
        except ClientRepositoryError as cre:
            assert str(cre) == "Id client inexistent."

        try:
            client_repository.delete_by_name(other_name)
            assert False
        except ClientRepositoryError as cre:
            assert str(cre) == "Nume client inexistent."

    @staticmethod
    def __test_filter_name_by_prefix_success(client_repository, client, prefix):
        filtered_list = client_repository.filter_name_by_prefix(prefix)
        assert filtered_list == [client]

    @staticmethod
    def __test_filter_name_by_non_existent_prefix(client_repositroy, prefix):
        try:
            client_repositroy.filter_name_by_prefix(prefix)
            assert False
        except ClientRepositoryError as cre:
            assert str(cre) == "Nu exista niciun client a carui nume sa inceapa cu acest prefix."

    def __run_tests_repository_client(self):
        client_repository = ClientsRepository()
        client_id = 1
        client_name = "Gabi"
        client_cnp = 5112233330000

        other_name = "Alex"

        non_existent_id = 2
        prefix = "Ga"

        client = Clients(client_id, client_name, client_cnp)
        other_client = Clients(client_id, other_name, client_cnp)

        self.__test_repository_add_client(client_repository, client)
        self.__test_repository_add_existing_client(client_repository, other_client)
        self.__test_repository_update_client_succes(client_repository, client_id, other_name)
        self.__test_repository_update_client_invalid(client_repository, non_existent_id, other_name)
        client = Clients(client_id, client_name, client_cnp)
        self.__test_repository_delete_client_by_existent_data(client_repository, client, client_id, client_name,
                                                              client_cnp)
        self.__test_repository_delete_client_by_non_existent_data(client_repository, non_existent_id, other_name)
        self.__test_filter_name_by_prefix_success(client_repository, client, prefix)

    @staticmethod
    def __test_service_add_client_success(client_service, client_id, client_name, client_cnp):
        assert len(client_service.get_all_clients()) == 0

        client_service.add_client(client_id, client_name, client_cnp)
        clients = client_service.get_all_clients()
        assert len(clients) == 1
        assert clients[0].id == client_id
        assert clients[0].cnp == client_cnp

    @staticmethod
    def __test_service_add_client_invalid(client_service, invalid_client_id, invalid_client_name, invalid_client_cnp):
        try:
            client_service.add_client(invalid_client_id, invalid_client_name, invalid_client_cnp)
            assert False
        except ClientValidationError as cve:
            assert str(cve) == "Id-ul introdus este invalid.\nNumele introdus este invalid.\n" \
                               "Cnp-ul introdus este invalid.\n"

    @staticmethod
    def __test_service_delete_client_success(client_service, client_id, client_name, client_cnp):
        client_service.delete_client_by_id(client_id)
        assert len(client_service.get_all_clients()) == 0
        client_service.add_client(client_id, client_name, client_cnp)

        client_service.delete_client_by_cnp(client_cnp)
        assert len(client_service.get_all_clients()) == 0
        client_service.add_client(client_id, client_name, client_cnp)

    @staticmethod
    def __test_service_delete_client_invalid(client_service, invalid_client_id, invalid_client_name):
        try:
            client_service.delete_client_by_id(invalid_client_id)
            assert False
        except ClientValidationError as cve:
            assert str(cve) == "Id-ul introdus este invalid."

        try:
            client_service.delete_client_by_name(invalid_client_name)
            assert False
        except ClientValidationError as cve:
            assert str(cve) == "Numele introdus este invalid."

    @staticmethod
    def __test_service_delete_client_non_existent(client_service, non_existent_client_id, non_existent_name,
                                                  non_existent_cnp):
        try:
            client_service.delete_client_by_id(non_existent_client_id)
            assert False
        except ClientRepositoryError as cre:
            assert str(cre) == "Id client inexistent."

        try:
            client_service.delete_client_by_name(non_existent_name)
            assert False
        except ClientRepositoryError as cre:
            assert str(cre) == "Nume client inexistent."

        try:
            client_service.delete_client_by_cnp(non_existent_cnp)
            assert False
        except ClientRepositoryError as cre:
            assert str(cre) == "Cnp client inexistent."

    @staticmethod
    def __test_service_update_client_succes(client_service, client_id, other_name):
        client_service.update_client_name(client_id, other_name)
        clients = client_service.get_all_clients()
        assert clients[0].name == other_name

    @staticmethod
    def __test_service_update_client_invalid(client_service, client_id, invalid_client_id, invalid_client_name):
        try:
            client_service.update_client_name(invalid_client_id, invalid_client_name)
            assert False
        except ClientValidationError as cve:
            assert str(cve) == "Id-ul introdus este invalid."

        try:
            client_service.update_client_name(client_id, invalid_client_name)
            assert False
        except ClientValidationError as cve:
            assert str(cve) == "Numele introdus este invalid."

    @staticmethod
    def __test_service_update_client_non_existent(client_service, non_existent_client_id, non_existent_name):
        try:
            client_service.update_client_name(non_existent_client_id, non_existent_name)
            assert False
        except ClientRepositoryError as cre:
            assert str(cre) == "Id client inexistent."

    def __run_tests_service_client(self):
        client_repository = ClientsFileRepository("Save_files/test.txt")
        client_validator = ClientsValidator()
        client_service = ClientsService(client_repository, client_validator)
        client_id = 1
        client_name = "Gabi"
        client_cnp = 5112233330000

        other_name = "Alex"

        non_existent_client_id = 2
        non_existent_name = "Andrei"
        non_existent_cnp = 8993344551234

        invalid_client_id = -1
        invalid_client_name = ""
        invalid_client_cnp = -1223344551234

        self.__test_service_add_client_success(client_service, client_id, client_name, client_cnp)
        self.__test_service_add_client_invalid(client_service, invalid_client_id, invalid_client_name, invalid_client_cnp)
        self.__test_service_delete_client_success(client_service, client_id, client_name, client_cnp)
        self.__test_service_delete_client_invalid(client_service, invalid_client_id, invalid_client_name)
        self.__test_service_delete_client_non_existent(client_service, non_existent_client_id, non_existent_name,
                                                       non_existent_cnp)
        self.__test_service_update_client_succes(client_service, client_id, other_name)
        self.__test_service_update_client_invalid(client_service, client_id, invalid_client_id, invalid_client_name)
        self.__test_service_update_client_non_existent(client_service, non_existent_client_id, non_existent_name)

    @staticmethod
    def __test_create_rent(book, client, date):
        rent = RentDto(book.id, client.id, date)
        rent.book = book
        rent.client = client
        assert rent.book.id == 1
        assert rent.date == "18/11/2021"
        assert rent.client.cnp == 1122334455667
        return rent

    @staticmethod
    def __test_equal_rent(rent1, book, client2, date):
        rent2 = RentDto(book.id, client2.id, date)
        assert rent1 == rent2

    @staticmethod
    def __test_print_rent(rent):
        assert str(rent) == "Clientul 'Mogage' a inchiriat cartea 'Ion' scrisa de 'Rebreanu' la data de 18/11/2021"

    def _run_tests_create_rent(self):
        book = Books(1, "Ion", "Pamant", "Rebreanu")
        client = Clients(1, "Mogage", 1122334455667)
        date = "18/11/2021"

        client2 = Clients(1, "Nicolae", 22334455667)

        rent1 = self.__test_create_rent(book, client, date)
        self.__test_equal_rent(rent1, book, client2, date)
        self.__test_print_rent(rent1)

    @staticmethod
    def __test_repository_add_rent(rent_repository, rent):
        assert len(rent_repository) == 0

        rent_repository.add_rent(rent)
        assert len(rent_repository) == 1
        assert rent_repository.rent[0].book.id == 1
        assert rent_repository.rent[0].book.author == "Rebreanu"
        assert rent_repository.rent[0].client.cnp == 1122334455667
        assert rent_repository.rent[0].date == "18/11/2021"

    @staticmethod
    def __test_repository_add_existing_rent(rent_repository, other_rent):
        try:
            rent_repository.add_rent(other_rent)
            assert False
        except RentRepositoryError as rre:
            assert str(rre) == "Acesta carte este deja inchiriata de acest client."

    @staticmethod
    def __test_repository_delete_rent_success(rent_repository, rent):
        rent_repository.delete_rent(rent.book.id, rent.client.id)
        assert len(rent_repository) == 0
        rent_repository.add_rent(rent)

    def __run_tests_repository_rent(self):
        rent_repository = RentFileRepository("Save_files/test.txt")
        book = Books(1, "Ion", "Pamant", "Rebreanu")
        client = Clients(1, "Mogage", 1122334455667)
        date = "18/11/2021"
        rent = RentDto(book.id, client.id, date)
        rent.book = book
        rent.client = client

        other_client = Clients(1, "Nicolae", 22334455667)
        other_rent = RentDto(book.id, other_client.id, date)

        self.__test_repository_add_rent(rent_repository, rent)
        self.__test_repository_add_existing_rent(rent_repository, other_rent)
        self.__test_repository_delete_rent_success(rent_repository, rent)

    @staticmethod
    def __test_service_add_rent_succes(rent_service):
        assert len(rent_service.get_all_rent()) == 0

        date = datetime.datetime.now().strftime("%d/%m/%y")
        rent_service.add_rent(1, 1)
        rent = rent_service.get_all_rent()
        assert rent[0].book.id == 1
        assert rent[0].book.title == "Ion"
        assert rent[0].client.name == "Mogage"
        assert rent[0].date == date

    @staticmethod
    def __test_service_add_rent_invalid(rent_service):
        try:
            rent_service.add_rent(-2, 5)
            assert False
        except RentValidationError as rve:
            assert str(rve) == "Id-ul introdus este invalid."
        try:
            rent_service.add_rent(1, -5)
            assert False
        except RentValidationError as rve:
            assert str(rve) == "Id-ul introdus este invalid."

    @staticmethod
    def __test_service_add_rent_non_existent(rent_service):
        try:
            rent_service.add_rent(1, 5)
            assert False
        except ClientRepositoryError as cre:
            assert str(cre) == "Id client inexistent."
        try:
            rent_service.add_rent(5, 1)
            assert False
        except BookRepositoryError as bre:
            assert str(bre) == "Id carte inexistent."

    @staticmethod
    def __test_service_delete_rent_succes(rent_service):
        rent_service.delete_rent(1, 1)
        assert len(rent_service.get_all_rent()) == 0
        rent_service.add_rent(1, 1)

    @staticmethod
    def __test_service_delete_rent_invalid(rent_service):
        try:
            rent_service.delete_rent(-1, 1)
            assert False
        except RentValidationError as rve:
            assert str(rve) == "Id-ul introdus este invalid."
        try:
            rent_service.delete_rent(1, -1)
            assert False
        except RentValidationError as rve:
            assert str(rve) == "Id-ul introdus este invalid."

    @staticmethod
    def __test_service_delete_rent_non_existent(rent_service):
        try:
            rent_service.delete_rent(5, 1)
            assert False
        except BookRepositoryError as bre:
            assert str(bre) == "Id carte inexistent."
        try:
            rent_service.delete_rent(1, 5)
            assert False
        except ClientRepositoryError as cre:
            assert str(cre) == "Id client inexistent."
        try:
            rent_service.delete_rent(2, 1)
            assert False
        except RentRepositoryError as rre:
            assert str(rre) == "Acest client nu a inchiriat aceasta carte."
        try:
            rent_service.delete_rent(1, 2)
            assert False
        except RentRepositoryError as rre:
            assert str(rre) == "Acest client nu are inchiriata nicio carte in acest moment."
    
    def __run_tests_service_rent(self):
        with open("Save_files/test.txt", "w"):
            pass
        books_validator = BooksValidator()
        clients_validator = ClientsValidator()
        rent_validator = RentValidator()

        books_repository = BooksFileRepository("Save_files/test.txt")
        clients_repository = ClientsFileRepository("Save_files/test.txt")
        rent_repository = RentFileRepository("Save_files/test.txt")

        books_service = BooksService(books_repository, books_validator)
        clients_service = ClientsService(clients_repository, clients_validator)
        rent_service = RentService(rent_repository, rent_validator, books_service, clients_service)

        books_service.add_book(1, "Ion", "Pamant", "Rebreanu")
        books_service.add_book(2, "Ion", "Pamant", "Rebreanu")
        clients_service.add_client(1, "Mogage", 1122334455667)
        clients_service.add_client(2, "Mogage", 1122334455668)

        self.__test_service_add_rent_succes(rent_service)
        self.__test_service_add_rent_invalid(rent_service)
        self.__test_service_add_rent_non_existent(rent_service)
        self.__test_service_delete_rent_succes(rent_service)
        self.__test_service_delete_rent_invalid(rent_service)
        self.__test_service_delete_rent_non_existent(rent_service)

    def run_all_tests(self):
        print("Start tests...")
        with open("Save_files/test.txt", "w"):
            pass
        self.__run_tests_create_book()
        self.__run_tests_validate_book()
        self.__run_tests_repository_book()
        self.__run_tests_service_book()

        with open("Save_files/test.txt", "w"):
            pass
        self.__run_tests_create_client()
        self.__run_tests_validate_client()
        self.__run_tests_repository_client()
        self.__run_tests_service_client()

        with open("Save_files/test.txt", "w"):
            pass
        self._run_tests_create_rent()
        self.__run_tests_repository_rent()
        self.__run_tests_service_rent()
        print("Finish tests...")
