#pragma once

#include <string>

class GeneralExceptions
{
private:
	std::string Message;
public:
	GeneralExceptions(std::string Message) : Message{ Message } {}
	const std::string& getMessage() const { return Message; }
};

class ValidationError : public GeneralExceptions
{
public:
	ValidationError(std::string Message) : GeneralExceptions(Message) {}
};

class RepositoryError : public GeneralExceptions
{
public:
	RepositoryError(std::string Message) : GeneralExceptions(Message) {}
};
