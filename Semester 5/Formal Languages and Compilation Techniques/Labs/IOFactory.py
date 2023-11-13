from finiteAutomata import FA


def read_content_from_file(file_name):
    with open(file_name) as file:
        states = file.readline().replace('\n', '').split(' ')
        alphabet = file.readline().replace('\n', '').split(' ')
        begin_states = file.readline().replace('\n', '').split(' ')
        end_states = file.readline().replace('\n', '').split(' ')
        lines = file.readlines()
        dictionary = {}

        for line in lines:
            items = line.replace('\n', '').split(',')
            if items[0] + '-' + items[2] not in dictionary.keys():
                dictionary[items[0] + '-' + items[2]] = []
            dictionary[items[0] + '-' + items[2]].append(items[1])

    return FA(states, alphabet, begin_states, end_states, dictionary)


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

    return FA(states, alphabet, begin_states, end_states, dictionary)


