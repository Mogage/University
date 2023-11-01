def read_content_from_file(file_name):
    with open(file_name) as file:
        states = file.readline().replace('\n', '').split(' ')
        alphabet = file.readline().replace('\n', '').split(' ')
        begin_states = file.readline().replace('\n', '').split(' ')
        end_states = file.readline().replace('\n', '').split(' ')
        lines = file.readlines()
        dictionary = {}

        for line in lines:
            items = line.replace('\n', '').split(' ')
            if items[0] + '-' + items[2] not in dictionary.keys():
                dictionary[items[0] + '-' + items[2]] = []
            dictionary[items[0] + '-' + items[2]].append(items[1])

    return states, alphabet, begin_states, end_states, dictionary


def read_content_from_keyboard():
    states = input("Enter the states: ").split(' ')
    alphabet = input("Enter the alphabet: ").split(' ')
    begin_states = input("Enter the begin states: ").split(' ')
    end_states = input("Enter the end states: ").split(' ')
    number_of_transitions = int(input("Enter the number of transitions: "))

    dictionary = {}

    for i in range(number_of_transitions):
        transition = input("Enter the transition: ").split(' ')
        if transition[0] + '-' + transition[2] not in dictionary.keys():
            dictionary[transition[0] + '-' + transition[2]] = []
        dictionary[transition[0] + '-' + transition[2]].append(transition[1])

    return states, alphabet, begin_states, end_states, dictionary


def print_response(current_states, prefix):
    for state in current_states:
        if state in end_states:
            print("The sequence is accepted.")
            break
    else:
        prefix = prefix[:-1]
        print("The sequence is not accepted.")

    if len(prefix) == 0:
        if begin_states[0] in end_states:
            print("Epsilon")
        else:
            print("The prefix is empty.")
    else:
        print('The prefix is: ' + prefix)


def solve_sequence(sequence):
    current_states = begin_states
    prefix = ''

    for character in sequence:
        next_states = []
        if len(current_states) == 0:
            break
        for state in current_states:
            for key, value in dictionary.items():
                if key.find(state) == 0 and character in value:
                    next_states.append(key[key.find('-') + 1:])
        prefix += character
        current_states = next_states

    print_response(current_states, prefix)


def print_menu():
    print("1. Show states")
    print("2. Show alphabet")
    print("3. Show begin states")
    print("4. Show end states")
    print("5. Show transitions")
    print("6. Solve a sequence.")
    print("0. Exit.")


if __name__ == '__main__':
    inputType = int(input('Read the data from: 1.File or 2. Keyboard.\n>>'))
    if inputType == 1:
        file_name = input('Input file: ')
        states, alphabet, begin_states, end_states, dictionary = read_content_from_file(file_name)
    else:
        states, alphabet, begin_states, end_states, dictionary = read_content_from_keyboard()

    while True:
        print_menu()
        option = int(input('>>'))

        if option == 0:
            break

        if option == 1:
            print(states)
        elif option == 2:
            print(alphabet)
        elif option == 3:
            print(begin_states)
        elif option == 4:
            print(end_states)
        elif option == 5:
            print(dictionary)
        elif option == 6:
            sequence = input("Enter the sequence: ")
            solve_sequence(sequence)
