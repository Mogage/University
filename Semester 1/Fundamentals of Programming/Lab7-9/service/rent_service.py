"""
    Author: Mogage Nicolae
    Creation date: 22 nov 2021
    Modul pentru gestiunea entitatilor inchiriere
"""
import datetime
import random

from domain.dtos import RentDto
from error.errors import RentRepositoryError

class RentService:
    """
        Clasa pentru gestiunea entitatilor inchiriere
    """
    def __init__(self, rent_repository, rent_validator, books_repository, clients_repository):
        """
            Initializarea datelor pentru gestiunea inchirierilor
        :param rent_repository: repo inchirieri
        :param rent_validator: validator inchirieri
        :param books_repository: repo carti
        :param clients_repository: repo clienti
        """
        self.__books_repository = books_repository
        self.__clients_repository = clients_repository
        self.__rent_repository = rent_repository
        self.__rent_validator = rent_validator
        self.__create_entities()
    
    def __create_entities(self):
        """
            Initializeaza entitatile dupa id-ul din dto
        """
        index = 0
        length = len(self.__rent_repository)
        while index < length:
            self.__rent_repository.rent[index].client = self.__clients_repository.find_client_by_id(self.__rent_repository.rent[index].id_client)
            self.__rent_repository.rent[index].book = self.__books_repository.find_book_by_id(self.__rent_repository.rent[index].id_book)
            index = index + 1
    
    def check_book_exists(self, id):
        """
            Verifica daca o carte exista
        :param id: int
        """
        self.__rent_validator.validate_id(id)
        self.__books_repository.find_book_by_id(id)

    def check_client_exists(self, id):
        """
            Verifica daca un client exista
        :param id: int
        """
        self.__rent_validator.validate_id(id)
        self.__clients_repository.find_client_by_id(id)

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
        self.__rent_validator.validate_id(id_book)
        self.__rent_validator.validate_id(id_client)
        date = datetime.datetime.now().strftime("%d/%m/%y")
        rent = RentDto(id_book, id_client, date)
        rent.book = self.__books_repository.find_book_by_id(id_book)
        rent.client = self.__clients_repository.find_client_by_id(id_client)
        self.__rent_repository.add_rent(rent)

    def __generate_data(self):
        """
            Genereaza datele pentru o entitate inchiriere
        :return: int, int, date - id_book, id_client, data in dd/mm/yyyy
        """
        index_book = random.randint(0, len(self.__books_repository) - 1)
        index_client = random.randint(0, len(self.__clients_repository) - 1)

        book = self.__books_repository.books[index_book]
        client = self.__clients_repository.clients[index_client]

        start_date = datetime.date(2000, 1, 1)
        end_date = datetime.date(2019, 12, 31)
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        date = start_date + datetime.timedelta(days=random_number_of_days)

        return book, client, date.strftime("%d/%m/%y")

    def generate_rent(self, number_of_rent):
        """
            Genereaza number_of_rent entitati de tip rent
        :param number_of_rent: numarul de inchirieri care sa se genereze
        """
        index = 0
        while index < number_of_rent:
            index = index + 1
            book, client, date = self.__generate_data()
            rent = RentDto(book.id, client.id, date)
            rent.book = book
            rent.client = client
            try:
                self.__rent_repository.add_rent(rent)
            except RentRepositoryError as rre:
                print(rre)
                continue
            print("Inchiriere efectuata cu succes.")

    def delete_rent(self, id_book, id_client):
        """
            Daca clinentul cu id-ul id_client a inchiriat cartea cu id-ul id_book sterge aceasta inchiriere din repo
        :param id_book: int
        :param id_client: int
        """
        self.check_rent_client_book_exists(id_book, id_client)
        self.__rent_repository.delete_rent(id_book, id_client)

    def __sort_rent_times(self, rent_times):
        """
            Sorteaza un dictionar {id_book: rent_times} dupa rent_times descrescator intr-o lista in care
            adauga cartea cu id-ul id_book
        :param rent_times: dictionary {} - dictionar in care apare de cate ori o carte a fost inchiriata
        :return: list [] - lista ordonata descrescator
        """
        sorted_rent_times = []
        number_of_rent = 0
        for key, value in rent_times.items():
            index = 0
            while index < number_of_rent and value < sorted_rent_times[index]["rent_times"]:
                index = index + 1

            number_of_rent = number_of_rent + 1
            book = self.__books_repository.find_book_by_id(key)
            query_to_print = {"book": book, "rent_times": value}
            sorted_rent_times.insert(index, query_to_print)

        return sorted_rent_times

    def __query_most_rented_books(self):
        """
            Calculeaza de cate ori este inchiriata fiecare carte
        :return: dictionary {} - dictionar {id_book, rent_times} in care cartea cu id-ul id_book
                                a fost inchiriata de rent_times
        """
        rent_times = {}
        for _rent in self.__rent_repository.rent:
            if _rent.id_book in rent_times.keys():
                rent_times[_rent.id_book] = rent_times[_rent.id_book] + 1
            else:
                rent_times[_rent.id_book] = 1

        return rent_times

    def print_most_rented_books(self):
        """
            Afiseaza cartile in ordine descrescatoare dupa numarul de inchirieri
        """
        rent_times = self.__query_most_rented_books()
        sorted_rent_times = self.__sort_rent_times(rent_times)
        for _rent_times in sorted_rent_times:
            print("Cartea", _rent_times["book"].title, "scrisa de", _rent_times["book"].author + ".")

    def __sort_active_clients(self, number_of_books_rented):
        """
            Sorteaza un dictionar {id_client, number_of_books_rented} dupa number_of_books_rented
            descrescator si in caz de egalitate, crescator dupa numele clientului, intr-o lista in care
            se memoreaza clientul si numarul de carti inchiriate
        :param number_of_books_rented: dict {} - dictionar care a stocat numarul de carti inchiriate
                                                 de un client
        :return: list [] - lista ordonata dupa criterii
        """
        sorted_active_clients = []
        sorted_active_clients_length = 0
        for key, value in number_of_books_rented.items():
            index = 0
            client = self.__clients_repository.find_client_by_id(key)

            while index < sorted_active_clients_length and \
                    (value < sorted_active_clients[index]["number_of_books_rented"] or
                     (value == sorted_active_clients[index]["number_of_books_rented"] and
                      client.name > sorted_active_clients[index]["client"].name)):
                index = index + 1

            sorted_active_clients_length = sorted_active_clients_length + 1
            query_to_print = {"client": client, "number_of_books_rented": value}
            sorted_active_clients.insert(index, query_to_print)

        return sorted_active_clients

    def __query_most_active_clients(self):
        """
            Calculeaza cate carti a inchiriat un client
        :return: dictionary {} - dictionar {id_client, number_of_books_rented} in care
                                 clientul cu id-ul id_client a inchiriat number_of_books_rented carti
        """
        number_of_books_rented = {}
        for _rent in self.__rent_repository.rent:
            if _rent.id_client in number_of_books_rented.keys():
                number_of_books_rented[_rent.id_client] = number_of_books_rented[_rent.id_client] + 1
            else:
                number_of_books_rented[_rent.id_client] = 1

        return number_of_books_rented

    def print_most_active_clients(self, top=None):
        """
            Afiseaza clientii in ordine descrescatoare dupa numarul de carti inchiriate,
            iar in caz de egalitate, dupa nume, iar daca top!=None afiseaza primii 20%
        :param top: Bool
        """
        number_of_books_rented = self.__query_most_active_clients()
        sorted_active_clients = self.__sort_active_clients(number_of_books_rented)
        if top is not None:
            number_of_clients_with_rented_books = int(len(number_of_books_rented) * 0.2) + 1
            sorted_active_clients = sorted_active_clients[:number_of_clients_with_rented_books]
        for _client in sorted_active_clients:
            string_to_be_printed = "Clientul " + _client["client"].name + " a inchiriat "
            if _client["number_of_books_rented"] == 1:
                string_to_be_printed = string_to_be_printed + "o carte."
            else:
                string_to_be_printed = string_to_be_printed + str(_client["number_of_books_rented"])
                if _client["number_of_books_rented"] >= 20:
                    string_to_be_printed = string_to_be_printed + " de "
                string_to_be_printed = string_to_be_printed + " carti."

            print(string_to_be_printed)

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
        self.check_rent_client_exists(id)
        self.__rent_repository.print_rent_by_client_id(id)

    def get_all_rent(self):
        """
            Getter pentru toate inchirierile
        :return: list []
        """
        return self.__rent_repository.rent
