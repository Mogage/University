def list_to_string(param_list):
    string = ''
    for element in param_list:
        if element == ';':
            continue
        string += element
    return string