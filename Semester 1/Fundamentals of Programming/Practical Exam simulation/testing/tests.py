import unittest

from domain.data import Rent, Books
from repository.repository import BookFileRepository, RentFileRepository
from service.service import BookService, RentService


class TestsBook(unittest.TestCase):
    def setUp(self):
        self.id = 1
        self.title = "Nume1"
        self.author = "Autor1"
        self.year = 2000
        self.book = Books(self.id, self.title, self.author, self.year)
        self.repository = BookFileRepository("Files/test.txt")

    def tearDown(self):
        pass

    def test_create_book(self):
        self.assertEqual(self.book.id, self.id)
        self.assertEqual(self.book.title, self.title)
        self.assertEqual(self.book.author, self.author)
        self.assertEqual(self.book.year, self.year)

    def test_load_file(self):

        self.assertEqual(len(self.repository.books), 3)
        self.assertEqual(self.repository.books[0], self.book)

    def test_requirement_1(self):
        self.service = BookService(self.repository)
        filtered_list = self.service.requirement_1("ceva")
        self.assertEqual(len(filtered_list), 0)
        filtered_list = self.service.requirement_1("2")
        self.assertEqual(len(filtered_list), 1)


class TestsRent(unittest.TestCase):
    def setUp(self):
        self.id = 1
        self.id_book = 1
        self.date = "5.12.2021"
        self.duration = 13
        self.rent = Rent(self.id, self.id_book, self.date, self.duration)
        self.repository = RentFileRepository("Files/test_rent.txt")

    def tearDown(self):
        pass

    def test_create_book(self):
        self.assertEqual(self.rent.id, self.id)
        self.assertEqual(self.rent.id_book, self.id_book)
        self.assertEqual(self.rent.date, self.date)
        self.assertEqual(self.rent.duration, self.duration)

    def test_load_file(self):
        self.assertEqual(len(self.repository.rents), 2)
        self.assertEqual(self.repository.rents[0], self.rent)

    def test_requirement_2(self):
        self.service = RentService(self.repository)
        filtered_list = self.service.requirement_2(25)
        self.assertEqual(len(filtered_list), 0)
        filtered_list = self.service.requirement_2(13)
        self.assertEqual(len(filtered_list), 1)


def run_tests():
    unittest.main(module=__name__, exit=False)
