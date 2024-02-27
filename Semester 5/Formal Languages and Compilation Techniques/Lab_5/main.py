from grammar import Grammar
from parser import Parser
from utils import list_to_string


if __name__ == '__main__':
    grammar = Grammar()
    # grammar.create_grammar('grammar.txt')
    grammar.create_grammar('LLGrammar.txt')

    # print(grammar)

    parser = Parser(grammar)

    with open("verify.txt", "r") as file:
        program = file.read()
    output, output_sequence = parser.parse(program)
    # output, output_sequence = parser.parse('bipie')

    if output[-1] == 'ERROR':
        print("Sequence is not accepted.")
        exit(0)

    print(output)
    print(output_sequence)

    print("Succes")

    string = grammar.start_symbol
    aux_string = string
    for index in output_sequence:
        non_terminal, expansion = grammar.get_by_index(index - 1)
        aux_string = aux_string.replace(non_terminal, list_to_string(expansion), 1)
        string += f' {index}-> {aux_string}'

    # print(string)
