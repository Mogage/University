from IOFactory import read_content_from_file, read_content_from_keyboard
from codeTransformer import CodeTransformer


def print_menu():
    print("1. Show states")
    print("2. Show alphabet")
    print("3. Show begin states")
    print("4. Show end states")
    print("5. Show transitions")
    print("6. Solve a sequence.")
    print("0. Exit.")


def handle_option(option, fa):
    if option == 1:
        print(fa.states)
    elif option == 2:
        print(fa.alphabet)
    elif option == 3:
        print(fa.begin_states)
    elif option == 4:
        print(fa.end_states)
    elif option == 5:
        print(fa.dictionary)
    elif option == 6:
        sequence = input("Enter the sequence: ")
        fa.get_longest_prefix(sequence)


def verify_automata_file():
    input_type = int(input('Read the data from: 1.File or 2. Keyboard.\n>>'))
    if input_type == 1:
        file_name = input('Input file: ')
        fa = read_content_from_file(file_name)
    else:
        fa = read_content_from_keyboard()

    while True:
        print_menu()
        option = int(input('>>'))

        if option == 0:
            break

        handle_option(option, fa)


if __name__ == '__main__':
    file_name = 'Exercise_3.cpp'
    codeTransformer = CodeTransformer()
    codeTransformer.transform_code(file_name)
