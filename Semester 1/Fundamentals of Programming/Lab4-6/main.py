from test import run_tests
from UI.UI_menu_driven import ui_user
from UI import UI_command_driven

if __name__ == "__main__":
    run_tests()
    ui_user()
    """read_input = UI_command_driven.UiMenu()
    read_input.ui_user()"""
    print("The program has stopped.")
