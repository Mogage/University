class FA:
    def __init__(self, states, alphabet, begin_states, end_states, dictionary):
        self.states = states
        self.alphabet = alphabet
        self.begin_states = begin_states
        self.end_states = end_states
        self.dictionary = dictionary

    def interpret_response(self, current_states, prefix):
        for state in current_states:
            if state in self.end_states:
                # print("The sequence is accepted.")
                break
        else:
            prefix = prefix[:-1]
            # print("The sequence is not accepted.")

        # if len(prefix) == 0:
        #     if self.begin_states[0] in self.end_states:
        #         print("Epsilon")
        #     else:
        #         print("The prefix is empty.")
        # else:
        #     print('The prefix is: ' + prefix)

        return prefix

    def get_longest_prefix(self, sequence_to_solve):
        current_states = self.begin_states
        prefix = ''

        for character in sequence_to_solve:
            next_states = []
            if len(current_states) == 0:
                break
            for state in current_states:
                for key, value in self.dictionary.items():
                    if key.find(state) == 0 and character in value:
                        next_states.append(key[key.find('-') + 1:])
            prefix += character
            current_states = next_states

        return self.interpret_response(current_states, prefix)
