"""
    Author: Mogage Nicolae
    Creation date: 24 nov 2021
    Modul pentru testarea functionalitatilor cu unittest
"""
import datetime
import unittest

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

class _TestCaseCreateBook(unittest.TestCase):
    def setUp(self):
        self.id_book = 1
        self.book_title = "Arhitectura calculatoarelor"
        self.book_description = "ASC stuff"
        self.book_author = "Alexandru Vancea"
        self.book = Books(self.id_book, self.book_title, self.book_description, self.book_author)

        self.other_title = "Fundamentele programarii"

    def tearDown(self):
        pass

    def testCreateBook(self):
        self.assertTrue(self.book.id == self.id_book)
        self.assertTrue(self.book.title == self.book_title)
        self.assertTrue(self.book.description == self.book_description)
        self.assertTrue(self.book.author == self.book_author)
        self.book.description = "Assembly"
        self.assertTrue(self.book.description == "Assembly")

    def test_equal_books(self):
        other_book = Books(self.id_book, self.other_title, self.book_description, self.book_author)
        self.assertTrue(other_book == self.book)

    def test_print_book(self):
        self.assertTrue(str(self.book) == "[1]Titlu: Arhitectura calculatoarelor\n"
                                          "Descriere: ASC stuff\n"
                                          "Autor: Alexandru Vancea")

class _TestCaseValidateBook(unittest.TestCase):
    def setUp(self):
        self.book_validator = BooksValidator()
        self.id_book = 1
        self.book_title = "Arhitectura calculatoarelor"
        self.book_description = "ASC stuff"
        self.book_author = "Alexandru Vancea"

        self.invalid_id_book = -1
        self.invalid_book_title = ""

    def tearDown(self):
        pass

    def test_validate_book_success(self):
        book = Books(self.id_book, self.book_title, self.book_description, self.book_author)
        try:
            self.book_validator.validate_book(book)
        except BookValidationError:
            self.fail("Validarea ar trebui sa fie cu succes.")

    def test_validate_book_invalid(self):
        book = Books(self.invalid_id_book, self.invalid_book_title, self.book_description, self.book_author)
        self.assertRaises(BookValidationError, self.book_validator.validate_book, book)

    def test_validate_book_specifications_success(self):
        try:
            self.book_validator.validate_id(self.id_book)
        except BookValidationError:
            self.fail("Validarea ar trebui sa fie cu succes.")

        try:
            self.book_validator.validate_author(self.book_author)
        except BookValidationError:
            self.fail("Validarea ar trebui sa fie cu succes.")

    def test_validate_book_specifications_invalid(self):
        self.assertRaises(BookValidationError, self.book_validator.validate_id, self.invalid_id_book)
        self.assertRaises(BookValidationError, self.book_validator.validate_title, self.invalid_book_title)

class _TestCaseRepositoryBook(unittest.TestCase):
    def setUp(self):
        self.book_repository = BooksRepository()
        self.id_book = 1
        self.non_existent_id_book = 5
        self.non_existent_book_title = "asc"
        self.book_title = "Arhitectura calculatoarelor"
        self.book_description = "ASC stuff"
        self.book_author = "Alexandru Vancea"
        self.book = Books(self.id_book, self.book_title, self.book_description, self.book_author)

        self.other_title = "Fundamentele programarii"
        self.other_description = "Python stuff"
        self.other_book = Books(self.id_book, self.other_title, self.other_description, self.book_author)

        self.book_repository.add_book(self.book)

    def tearDown(self):
        pass

    def test_repository_add_book(self):
        self.assertTrue(len(self.book_repository.books) == 1)
        self.assertTrue(self.book_repository.books[0].id == self.book.id)
        self.assertTrue(self.book_repository.books[0].title == self.book.title)
        self.assertTrue(self.book_repository.books[0].author == self.book.author)

    def test_repository_add_existent_id_book(self):
        self.assertRaises(BookRepositoryError, self.book_repository.add_book, self.other_book)

    def test_repository_update_book_success(self):
        self.book_repository.update_book_title(self.id_book, self.other_title)
        self.book_repository.update_book_description(self.id_book, self.other_description)
        self.assertTrue(self.book_repository.books[0].id == self.id_book)
        self.assertTrue(self.book_repository.books[0].title == self.other_title)
        self.assertTrue(self.book_repository.books[0].description == self.other_description)

    def test_repository_update_book_non_existent_id(self):
        self.assertRaises(BookRepositoryError, self.book_repository.update_book_title, self.non_existent_id_book, self.book_title)

    def test_find_book_by_existent_id(self):
        found_book = self.book_repository.find_book_by_id(self.id_book)
        self.assertTrue(self.book.id == found_book.id)
        self.assertTrue(self.book.title == found_book.title)
        self.assertTrue(self.book.description == found_book.description)
        self.assertTrue(self.book.author == found_book.author)

    def test_find_book_by_non_existent_id(self):
        self.assertRaises(BookRepositoryError, self.book_repository.find_book_by_id, self.non_existent_id_book)

    def test_repository_delete_book_by_existent_data(self):
        self.book_repository.delete_book_by_id(self.id_book)
        self.assertTrue(len(self.book_repository) == 0)
        self.book_repository.add_book(self.book)

        self.book_repository.delete_book_by_title(self.book_title)
        self.assertTrue(len(self.book_repository) == 0)
        self.book_repository.add_book(self.book)

        self.book_repository.delete_book_by_description(self.book_description)
        self.assertTrue(len(self.book_repository) == 0)
        self.book_repository.add_book(self.book)

        self.book_repository.delete_book_by_author(self.book_author)
        self.assertTrue(len(self.book_repository) == 0)

    def test_repository_delete_book_by_non_existent_data(self):
        self.assertRaises(BookRepositoryError, self.book_repository.delete_book_by_id, self.non_existent_id_book)
        self.assertRaises(BookRepositoryError,  self.book_repository.delete_book_by_title, self.non_existent_book_title)

class _TestCaseServiceBook(unittest.TestCase):
    def setUp(self):
        self.book_repository = BooksFileRepository("Save_files/test.txt")
        self.book_validator = BooksValidator()
        self.book_service = BooksService(self.book_repository, self.book_validator)
        self.id_book = 1
        self.book_title = "Arhitectura calculatoarelor"
        self.book_description = "ASC stuff"
        self.book_author = "Alexandru Vancea"

        self.other_title = "Fundamentele programarii"

        self.invalid_id_book = -1
        self.invalid_book_title = ""
        self.invalid_book_description = ""
        self.invalid_book_author = ""

        self.non_existent_id_book = 2
        self.non_existent_description = "Python"
        self.non_existent_author = "Gabi Mircea"

        self.book_service.add_book(self.id_book, self.book_title, self.book_description, self.book_author)

    def tearDown(self):
        with open("Save_files/test.txt", "w"):
            pass

    def test_service_add_book_success(self):
        books = self.book_service.get_all_books()
        self.assertTrue(len(books) == 1)
        self.assertTrue(books[0].id == self.id_book)
        self.assertTrue(books[0].author == self.book_author)

    def test_service_add_book_invalid(self):
        self.assertRaises(BookValidationError, self.book_service.add_book, self.invalid_id_book,
                          self.invalid_book_title, self.invalid_book_description, self.invalid_book_author)

    def test_service_delete_book_success(self):
        self.book_service.delete_book_by_id(self.id_book)
        self.assertTrue(len(self.book_service.get_all_books()) == 0)
        self.book_service.add_book(self.id_book, self.book_title, self.book_description, self.book_author)

        self.book_service.delete_book_by_author(self.book_author)
        self.assertTrue(len(self.book_service.get_all_books()) == 0)

    def test_service_delete_book_invalid(self):
        self.assertRaises(BookValidationError, self.book_service.delete_book_by_id, self.invalid_id_book)
        self.assertRaises(BookValidationError, self.book_service.delete_book_by_title, self.invalid_book_title)

    def test_service_delete_book_non_existent(self):
        self.assertRaises(BookRepositoryError, self.book_service.delete_book_by_id, self.non_existent_id_book)
        self.assertRaises(BookRepositoryError, self.book_service.delete_book_by_description, self.non_existent_description)
        self.assertRaises(BookRepositoryError, self.book_service.delete_book_by_author, self.non_existent_author)

    def test_service_update_book_succes(self):
        self.book_service.update_book_title(self.id_book, self.other_title)
        books = self.book_service.get_all_books()
        self.assertTrue(books[0].title == self.other_title)

    def test_service_update_book_invalid(self):
        self.assertRaises(BookValidationError, self.book_service.update_book_title, self.invalid_id_book, self.invalid_book_title)
        self.assertRaises(BookValidationError, self.book_service.update_book_title, self.id_book, self.invalid_book_title)

    def test_service_update_book_non_existent(self):
        self.assertRaises(BookRepositoryError, self.book_service.update_book_author, self.non_existent_id_book, self.non_existent_author)

class _TestCaseCreateClient(unittest.TestCase):
    def setUp(self):
        self.client_id = 1
        self.client_name = "Gabi"
        self.client_cnp = 5112233330000

        self.other_name = "Alexandru"

        self.client = Clients(self.client_id, self.client_name, self.client_cnp)

    def tearDown(self):
        pass

    def test_create_client(self):
        self.assertTrue(self.client.id == self.client_id)
        self.assertTrue(self.client.name == self.client_name)
        self.assertTrue(self.client.cnp == self.client_cnp)
        self.client.name = "Andrei"
        self.assertTrue(self.client.name == "Andrei")

    def test_equal_clients(self):
        other_client = Clients(self.client_id, self.other_name, self.client_cnp)
        self.assertTrue(self.client == other_client)

    def test_print_client(self):
        self.assertTrue(str(self.client) == "[1]Nume: Gabi. CNP: 5112233330000")

class _TestCaseValidationClient(unittest.TestCase):
    def setUp(self):
        self.client_validator = ClientsValidator()
        self.client_id = 1
        self.client_name = "Gabi"
        self.client_cnp = 5112233330000

        self.invalid_client_id = -1
        self.invalid_client_name = ""
        self.invalid_client_cnp = 25
        self.invalid_client_cnp_2 = -5112233330000

    def tearDown(self):
        pass

    def test_validate_client_success(self):
        client = Clients(self.client_id, self.client_name, self.client_cnp)
        try:
            self.client_validator.validate_client(client)
        except ClientValidationError:
            self.fail("Validarea ar trebui sa fie cu succes.")

    def test_validate_client_invalid(self):
        client = Clients(self.invalid_client_id, self.invalid_client_name, self.invalid_client_cnp)
        self.assertRaises(ClientValidationError, self.client_validator.validate_client, client)

    def test_validate_client_specifications_succes(self):
        try:
            self.client_validator.validate_id(self.client_id)
        except ClientValidationError:
            self.fail("Validarea ar trebui sa fie cu succes.")

        try:
            self.client_validator.validate_cnp(self.client_cnp)
        except ClientValidationError:
            self.fail("Validarea ar trebui sa fie cu succes.")

    def test_validate_client_specifications_invalid(self):
        self.assertRaises(ClientValidationError, self.client_validator.validate_name, self.invalid_client_name)
        self.assertRaises(ClientValidationError, self.client_validator.validate_cnp, self.invalid_client_cnp_2)

class _TestCaserepositoryClient(unittest.TestCase):
    def setUp(self):
        self.client_repository = ClientsRepository()
        self.client_id = 1
        self.client_name = "Gabi"
        self.client_cnp = 5112233330000

        self.other_name = "Alex"

        self.non_existent_id = 2
        self.prefix = "Ga"
        self.non_existent_prefix = "Al"

        self.client = Clients(self.client_id, self.client_name, self.client_cnp)
        self.other_client = Clients(self.client_id, self.other_name, self.client_cnp)

        self.client_repository.add_client(self.client)

    def tearDown(self):
        pass

    def test_repository_add_client(self):
        self.assertTrue(len(self.client_repository) == 1)
        self.assertTrue(self.client_repository.clients[0].id == self.client.id)
        self.assertTrue(self.client_repository.clients[0].name == self.client.name)
        self.assertTrue(self.client_repository.clients[0].cnp == self.client.cnp)

    def test_repository_add_existing_client(self):
        self.assertRaises(ClientRepositoryError, self.client_repository.add_client, self.other_client)

    def test_repository_update_client_succes(self):
        self.client_repository.update_name(self.client_id, self.other_name)
        self.assertTrue(self.client_repository.clients[0].name == self.other_name)

    def test_repository_update_client_invalid(self):
        self.assertRaises(ClientRepositoryError, self.client_repository.update_name, self.non_existent_id, self.other_name)
    
    def test_repository_delete_client_by_existent_data(self):
        self.client_repository.delete_by_id(self.client_id)
        self.assertTrue(len(self.client_repository) == 0)
        self.client_repository.add_client(self.client)

        self.client_repository.delete_by_name(self.client_name)
        self.assertTrue(len(self.client_repository) == 0)
        self.client_repository.add_client(self.client)

        self.client_repository.delete_by_cnp(self.client_cnp)
        self.assertTrue(len(self.client_repository) == 0)

    def test_repository_delete_client_by_non_existent_data(self):
        self.assertRaises(ClientRepositoryError, self.client_repository.delete_by_id, self.non_existent_id)
        self.assertRaises(ClientRepositoryError, self.client_repository.delete_by_name, self.other_name)

    def test_filter_name_by_prefix_success(self):
        filtered_list = self.client_repository.filter_name_by_prefix(self.prefix)
        self.assertIn(self.client, filtered_list)

    def test_filter_name_by_non_existent_prefix(self):
        self.assertRaises(ClientRepositoryError, self.client_repository.filter_name_by_prefix, self.non_existent_prefix)
    
class _TestCaseServiceClient(unittest.TestCase):
    def setUp(self):
        self.client_repository = ClientsFileRepository("Save_files/test.txt")
        self.client_validator = ClientsValidator()
        self.client_service = ClientsService(self.client_repository, self.client_validator)
        self.client_id = 1
        self.client_name = "Gabi"
        self.client_cnp = 5112233330000

        self.other_name = "Alex"

        self.non_existent_client_id = 2
        self.non_existent_name = "Andrei"
        self.non_existent_cnp = 8993344551234

        self.invalid_client_id = -1
        self.invalid_client_name = ""
        self.invalid_client_cnp = -1223344551234

        self.client_service.add_client(self.client_id, self.client_name, self.client_cnp)

    def tearDown(self):
        with open("Save_files/test.txt", "w"):
            pass

    def test_service_add_client_success(self):
        clients = self.client_service.get_all_clients()
        self.assertTrue(len(clients) == 1)
        self.assertTrue(clients[0].id == self.client_id)
        self.assertTrue(clients[0].cnp == self.client_cnp)

    def test_service_add_client_invalid(self):
        self.assertRaises(ClientValidationError, self.client_service.add_client, self.invalid_client_id,
                          self.invalid_client_name, self.invalid_client_cnp)

    def test_service_delete_client_success(self):
        self.client_service.delete_client_by_id(self.client_id)
        self.assertTrue(len(self.client_service.get_all_clients()) == 0)
        self.client_service.add_client(self.client_id, self.client_name, self.client_cnp)

        self.client_service.delete_client_by_cnp(self.client_cnp)
        self.assertTrue(len(self.client_service.get_all_clients()) == 0)

    def test_service_delete_client_invalid(self):
        self.assertRaises(ClientValidationError, self.client_service.delete_client_by_id, self.invalid_client_id)
        self.assertRaises(ClientValidationError, self.client_service.delete_client_by_name, self.invalid_client_name)

    def test_service_delete_client_non_existent(self):
        self.assertRaises(ClientRepositoryError, self.client_service.delete_client_by_id, self.non_existent_client_id)
        self.assertRaises(ClientRepositoryError, self.client_service.delete_client_by_name, self.non_existent_name)
        self.assertRaises(ClientRepositoryError, self.client_service.delete_client_by_cnp, self.non_existent_cnp)

    def test_service_update_client_succes(self):
        self.client_service.update_client_name(self.client_id, self.other_name)
        clients = self.client_service.get_all_clients()
        self.assertTrue(clients[0].name == self.other_name)

    def test_service_update_client_invalid(self):
        self.assertRaises(ClientValidationError, self.client_service.update_client_name, self.invalid_client_id,
                          self.invalid_client_name)
        self.assertRaises(ClientValidationError, self.client_service.update_client_name, self.client_id,
                          self.invalid_client_name)

    def test_service_update_client_non_existent(self):
        self.assertRaises(ClientRepositoryError, self.client_service.update_client_name, self.non_existent_client_id,
                          self.non_existent_name)

class _TestCaseCreateRent(unittest.TestCase):
    def setUp(self):
        self.book = Books(1, "Ion", "Pamant", "Rebreanu")
        self.client = Clients(1, "Mogage", 1122334455667)
        self.date = "18/11/2021"
    
        self.client2 = Clients(1, "Nicolae", 22334455667)

        self.rent = RentDto(self.book.id, self.client.id, self.date)
        self.rent.book = self.book
        self.rent.client = self.client

    def tearDown(self):
        pass

    def test_create_rent(self):
        self.assertTrue(self.rent.book.id == 1)
        self.assertTrue(self.rent.date == "18/11/2021")
        self.assertTrue(self.rent.client.cnp == 1122334455667)
        
    def test_equal_rent(self):
        rent2 = RentDto(self.book.id, self.client2.id, self.date)
        self.assertTrue(self.rent == rent2)

    def test_print_rent(self):
        self.assertTrue(str(self.rent) == "Clientul 'Mogage' a inchiriat cartea 'Ion' scrisa de 'Rebreanu'"
                                          " la data de 18/11/2021")

class _TestCaseRepositoryRent(unittest.TestCase):
    def setUp(self):
        self.rent_repository = RentFileRepository("Save_files/test.txt")
        self.book = Books(1, "Ion", "Pamant", "Rebreanu")
        self.client = Clients(1, "Mogage", 1122334455667)
        self.date = "18/11/2021"
        self.rent = RentDto(self.book.id, self.client.id, self.date)
        self.rent.book = self.book
        self.rent.client = self.client

        self.other_client = Clients(1, "Nicolae", 22334455667)
        self.other_rent = RentDto(self.book.id, self.other_client.id, self.date)

        self.rent_repository.add_rent(self.rent)

    def tearDown(self):
        with open("Save_files/test.txt", "w"):
            pass
        
    def test_repository_add_rent(self):
        self.assertTrue(len(self.rent_repository) == 1)
        self.assertTrue(self.rent_repository.rent[0].book.id == 1)
        self.assertTrue(self.rent_repository.rent[0].book.author == "Rebreanu")
        self.assertTrue(self.rent_repository.rent[0].client.cnp == 1122334455667)
        self.assertTrue(self.rent_repository.rent[0].date == "18/11/2021")

    def test_repository_add_existing_rent(self):
        self.assertRaises(RentRepositoryError, self.rent_repository.add_rent, self.other_rent)

    def test_repository_delete_rent_success(self):
        self.rent_repository.delete_rent(self.rent.book.id, self.rent.client.id)
        self.assertTrue(len(self.rent_repository) == 0)

class _TestCaseserviceRent(unittest.TestCase):
    def setUp(self):
        self.books_validator = BooksValidator()
        self.clients_validator = ClientsValidator()
        self.rent_validator = RentValidator()

        self.books_repository = BooksFileRepository("Save_files/test.txt")
        self.clients_repository = ClientsFileRepository("Save_files/test.txt")
        self.rent_repository = RentFileRepository("Save_files/test.txt")

        self.books_service = BooksService(self.books_repository, self.books_validator)
        self.clients_service = ClientsService(self.clients_repository, self.clients_validator)
        self.rent_service = RentService(self.rent_repository, self.rent_validator, self.books_service,
                                        self.clients_service)

        self.books_service.add_book(1, "Ion", "Pamant", "Rebreanu")
        self.books_service.add_book(2, "Ion", "Pamant", "Rebreanu")
        self.clients_service.add_client(1, "Mogage", 1122334455667)
        self.clients_service.add_client(2, "Mogage", 1122334455668)

        self.date = datetime.datetime.now().strftime("%d/%m/%y")
        self.rent_service.add_rent(1, 1)

    def tearDown(self):
        with open("Save_files/test.txt", "w"):
            pass

    def test_service_add_rent_succes(self):
        rent = self.rent_service.get_all_rent()
        self.assertTrue(rent[0].book.id == 1)
        self.assertTrue(rent[0].book.title == "Ion")
        self.assertTrue(rent[0].client.name == "Mogage")
        self.assertTrue(rent[0].date == self.date)

    def test_service_add_rent_invalid(self):
        self.assertRaises(RentValidationError, self.rent_service.add_rent, -2, 5)
        self.assertRaises(RentValidationError, self.rent_service.add_rent, 1, -5)

    def test_service_add_rent_non_existent(self):
        self.assertRaises(ClientRepositoryError, self.rent_service.add_rent, 1, 5)
        self.assertRaises(BookRepositoryError, self.rent_service.add_rent, 5, 1)

    def test_service_delete_rent_succes(self):
        self.rent_service.delete_rent(1, 1)
        self.assertTrue(len(self.rent_service.get_all_rent()) == 0)

    def test_service_delete_rent_invalid(self):
        self.assertRaises(RentValidationError, self.rent_service.delete_rent, -1, 1)
        self.assertRaises(RentValidationError, self.rent_service.delete_rent, 1, -1)

    def test_service_delete_rent_non_existent(self):
        self.assertRaises(BookRepositoryError, self.rent_service.delete_rent, 5, 1)
        self.assertRaises(ClientRepositoryError, self.rent_service.delete_rent, 1, 5)
        self.assertRaises(RentRepositoryError, self.rent_service.delete_rent, 2, 1)
        self.assertRaises(RentRepositoryError, self.rent_service.delete_rent, 1, 2)

def run_tests():
    unittest.main(module=__name__, exit=False)
