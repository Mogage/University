#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <WinSock2.h>
#include <WS2tcpip.h>

int main(int argc, char* argv[])
{
    SOCKET serverSocket;
    SOCKET clientSocket;
    SOCKADDR_IN serverInfo;
    SOCKADDR_IN clientInfo;
    CHAR string[101];
    UINT16 numberOfSpaces;
    UINT32 clientSize;
    WSADATA wsa;

    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
    {
        printf("WSAStartup failed.\n");
        exit(1);
    }

    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == INVALID_SOCKET)
    {
        printf("Error at creating server socket.\n");
        exit(1);
    }

    serverInfo.sin_family = AF_INET;
    serverInfo.sin_addr.s_addr = INADDR_ANY;
    serverInfo.sin_port = htons(1234);

    if (bind(serverSocket, (SOCKADDR*)&serverInfo, sizeof(SOCKADDR_IN)) < 0)
    {
        printf("error at bind.\n");
        exit(1);
    }

    listen(serverSocket, 5);

    clientSize = sizeof(clientInfo);

    printf("Server is up and running.\n");

    while (1)
    {
        numberOfSpaces = 0;
        clientSocket = accept(serverSocket, NULL, NULL);
        printf("Client connected.\n");

        recv(clientSocket, (char*)&string, sizeof(string), MSG_WAITALL);
        //strcpy(string, ntohs(string));
        for (int count = 0; string[count] != '\0'; count = count + 1)
        {
            if (' ' == string[count])
                numberOfSpaces = numberOfSpaces + 1;
        }
        numberOfSpaces = htons(numberOfSpaces);
        send(clientSocket, (const char*)&numberOfSpaces, sizeof(numberOfSpaces), 0);
        closesocket(clientSocket);
    }

    closesocket(serverSocket);
    return 0;
}