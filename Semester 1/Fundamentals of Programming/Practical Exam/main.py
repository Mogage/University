from repository import FileRepository
from validator import Validator
from service import Service
from user_interface import Ui


def run_app():
    """
        Functia main din care se controleaza toata aplicatia
    """
    file_path = "input.txt"

    event_repository = FileRepository(file_path)
    event_validator = Validator()
    event_service = Service(event_validator, event_repository)

    event_ui = Ui(event_service)

    event_ui.run_main()

run_app()
