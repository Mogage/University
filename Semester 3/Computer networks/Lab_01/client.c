#define _WINSOCK_DEPRECATED_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <WinSock2.h>
#include <WS2tcpip.h>

int main(int argc, char* argv[])
{
    SOCKET clientSocket;
    SOCKADDR_IN serverInfo;
    CHAR string[101] = { 0 };
    UINT16 numberOfSpaces;
    WSADATA wsa;

    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
    {
        printf("WSAStartup failed.\n");
        exit(1);
    }

    clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == INVALID_SOCKET)
    {
        printf("Error at creating server socket.\n");
        exit(1);
    }

    serverInfo.sin_family = AF_INET;
    serverInfo.sin_addr.s_addr = inet_addr("127.0.0.1");
    serverInfo.sin_port = htons(1234);

    if (connect(clientSocket, (SOCKADDR*)&serverInfo, sizeof(SOCKADDR_IN)) < 0)
    {
        printf("error at bind.\n");
        exit(1);
    }

    printf("String: ");
    fgets(string, 100, stdin);
    //strcpy(string, htons(string));
    send(clientSocket, (const char*)&string, sizeof(string), 0);
    recv(clientSocket, (char*) & numberOfSpaces, sizeof(numberOfSpaces), 0);
    numberOfSpaces = ntohs(numberOfSpaces);
    printf("Number of spaces = %hu.\n", numberOfSpaces);
    closesocket(clientSocket);
    return 0;
}