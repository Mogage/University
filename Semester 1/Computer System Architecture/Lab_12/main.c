#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>

// char* get_chars(char*, char, char, char*);
void get_chars(char*, char, char, char*);

int main()
{
	char* string = NULL;
	char* lower_case = NULL, * upper_case = NULL;
	string = (char*)malloc(101 * sizeof(char));
	lower_case = (char*)malloc(101 * sizeof(char));
	upper_case = (char*)malloc(101 * sizeof(char));
	if (string == NULL || lower_case == NULL || upper_case == NULL)
	{
		printf("Eroare la alocarea de memorie.");
		return -1;
	}
	
	if (scanf("%[^\n]s", string) == 0)
	{
		printf("Nu s-a putut citi de la tastatura.");
		return -1;
	}
	
	get_chars(string, 'a', 'z', lower_case);
	get_chars(string, 'A', 'Z', upper_case);

	printf("Acest sir contine urmatoarele litere mici: %s\n", lower_case);
	printf("Acest sir contine urmatoarele litere mari: %s\n", upper_case);

	free(string);
	free(lower_case);
	free(upper_case);
	string = NULL;
	return 0;
}

/*
char* get_chars(char* string, char lower_bound, char upper_bound)
{
	char* new_string = NULL;
	new_string = (char*)malloc(101 * sizeof(char));
	if (new_string == NULL)
	{
		printf("Eroare la alocarea de memorie.");
		return NULL;
	}
	int new_string_length = 0;
	for (int string_index = 0; *(string + string_index) != '\0'; string_index = string_index + 1)
	{
		if (*(string + string_index) >= lower_bound && *(string + string_index) <= upper_bound)
		{
			*(new_string + new_string_length) = *(string + string_index);
			new_string_length = new_string_length + 1;
		}
	}
	*(new_string + new_string_length) = '\0';
	printf("%d\n", new_string_length);
	return new_string;
}
*/