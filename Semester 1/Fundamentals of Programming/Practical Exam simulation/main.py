from repository.repository import BookFileRepository, RentFileRepository
from service.service import BookService, RentService
from user_interface.main_user_interface import Ui
from testing.tests import run_tests


if __name__ == "__main__":
    run_tests()

    books_path = "Files/books.txt"
    rent_path = "Files/rent.txt"
    books_repository = BookFileRepository(books_path)
    rent_repository = RentFileRepository(rent_path)

    books_service = BookService(books_repository)
    rent_service = RentService(rent_repository)

    main_ui = Ui(books_service, rent_service)
    main_ui.run()
