from testing.tests import Tests
from ui.main_user_interface import Console
from service.services import BooksService, ClientsService
from validator.validators import BooksValidator, ClientsValidator
from repository.repositories import BooksRepository, ClientsRepository

if __name__ == "__main__":
    books_validator = BooksValidator()
    clients_validator = ClientsValidator()

    books_repository = BooksRepository()
    clients_repository = ClientsRepository()

    books_service = BooksService(books_repository, books_validator)
    clients_service = ClientsService(clients_repository, clients_validator)

    ui = Console(books_service, clients_service)
    tests = Tests()
    tests.run_all_tests()

    ui.run()
