"""
    Author: Mogage Nicolae
    Creation date: 22 nov 2021
    Modul pentru gestiunea entitatilor client
"""
import random
import string

from domain.data import Clients

class ClientsService:
    """
        Clasa ce se ocupa gestiunea entitatilor de tip client
    """
    def __init__(self, clients_repository, clients_validator):
        """
            Initializarea unui serviciu de gestiune a clientilor
        :param clients_repository: repozitoriu pentru clienti
        :param clients_validator: validator pentru clienti
        """
        self.__clients_repository = clients_repository
        self.__clients_validator = clients_validator

    def add_client(self, id, name, cnp):
        """
            Creeaza, valideaza si adauga un client la lista
        :param id: int
        :param name: string
        :param cnp: int
        """
        client = Clients(id, name, cnp)
        self.__clients_validator.validate_client(client)
        self.__clients_repository.add_client(client)

    @staticmethod
    def __generate_data():
        """
            Genereaza datele pentru entitate client
        :return: int, string, int
        """
        id = random.randint(0, 1000)
        name = ''.join(random.choice(string.ascii_letters) for _ in range(id % 50))
        cnp = random.randint(1000000000000, 9999999999999)
        return id, name, cnp

    def generate_clients(self, number_of_clients):
        """
            Genereaza number_of_clients entitati de tip client
        :param number_of_clients: int - numarul de clienti care sa se genereze
        """
        index = 0
        while index < number_of_clients:
            id, name, cnp = self.__generate_data()
            client = Clients(id, name, cnp)
            self.__clients_validator.validate_client(client)
            self.__clients_repository.add_client(client)
            print("Client adaugat cu succes.")
            index = index + 1

    def update_client_name(self, id, name):
        """
            Valideaza id-ul si verifica daca exista un client cu acest id
            Valideaza numele si actualizeaza numele cartii
        :param id: int
        :param name: string
        """
        self.__clients_validator.validate_id(id)
        self.__clients_repository.find_client_by_id(id)
        self.__clients_validator.validate_name(name)
        self.__clients_repository.update_name(id, name)

    def delete_client_by_id(self, id):
        """
            Valideaza id-ul si sterge clientul cu acest id
        :param id: int
        """
        self.__clients_validator.validate_id(id)
        self.__clients_repository.delete_by_id(id)

    def delete_client_by_name(self, name):
        """
            Valideaza numele si sterge clientii cu acest nume
        :param name: string
        """
        self.__clients_validator.validate_name(name)
        self.__clients_repository.delete_by_name(name)

    def delete_client_by_cnp(self, cnp):
        """
            Valideaza cnp-ul si sterge clientul cu acest cnp
        :param cnp: int
        """
        self.__clients_validator.validate_cnp(cnp)
        self.__clients_repository.delete_by_cnp(cnp)

    def print_all_clients(self):
        """
            Afiseaza toti clientii
        """
        self.__clients_repository.print_all_clients()

    def print_clients_id(self, id):
        """
            Valideaza id-ul si afiseaza clientul cu acest id
        :param id: int
        """
        self.__clients_validator.validate_id(id)
        self.__clients_repository.print_clients_id(id)

    def print_clients_name(self, name):
        """
            Valideaza numele si afiseaza clientii cu acest nume
        :param name: string
        """
        self.__clients_validator.validate_name(name)
        self.__clients_repository.print_clients_name(name)

    def print_clients_cnp(self, cnp):
        """
            Valideaza cnp-ul si afiseaza clientul cu acest cnp
        :param cnp: int
        """
        self.__clients_validator.validate_cnp(cnp)
        self.__clients_repository.print_clients_cnp(cnp)

    def print_filter_name_by_prefix(self, prefix):
        """
            Valideaza prefixul si filtreaza lista de clienti, afisand clientii a caror nume incepe cu acest prefix
        :param prefix: string
        """
        self.__clients_validator.validate_prefix(prefix)
        self.__clients_repository.print_filter_name_by_prefix(prefix)

    def find_client_by_id(self, id):
        """
            Valideaza id-ul si verifica daca exista un client cu acest id
        :param id: int
        """
        self.__clients_validator.validate_id(id)
        return self.__clients_repository.find_client_by_id(id)

    def get_all_clients(self):
        """
            Functie de getter a tuturor clientilor
        :return: lista cu toti clientii
        """
        return self.__clients_repository.clients
