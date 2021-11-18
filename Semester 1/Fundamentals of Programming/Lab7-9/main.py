from testing.tests import Tests
from ui.main_user_interface import Console
from service.services import BooksService, ClientsService, RentService
from validator.validators import BooksValidator, ClientsValidator, RentValidator
from repository.book_repository import BooksFileRepository
from repository.client_repository import ClientsFileRepository
from repository.rent_repository import RentFileRepository

if __name__ == "__main__":
    books_validator = BooksValidator()
    clients_validator = ClientsValidator()
    rent_validator = RentValidator()

    books_repository = BooksFileRepository("Save_files/book.txt")
    clients_repository = ClientsFileRepository("Save_files/client.txt")
    rent_repository = RentFileRepository("Save_files/rent.txt")

    books_service = BooksService(books_repository, books_validator)
    clients_service = ClientsService(clients_repository, clients_validator)
    rent_service = RentService(rent_repository, rent_validator, books_service, clients_service)

    tests = Tests()
    tests.run_all_tests()

    ui = Console(books_service, clients_service, rent_service)
    ui.run()
    print("The program has ended successfully.")
