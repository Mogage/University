#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <arpa/inet.h>

typedef struct sockaddr_in SOCKADDR_IN;
typedef struct sockaddr SOCKADDR;
typedef uint16_t UINT16;

int main()
{
    int clientSocket;
    int isPrim;
    int serverSize;
    SOCKADDR_IN serverInfo;
    UINT16 number;

    clientSocket = socket(AF_INET, SOCK_DGRAM, 0);
    if(clientSocket < 0)
    {
        printf("Error at creating socket");
        exit(1);
    }

    memset(&serverInfo, 0, sizeof(serverInfo));
    serverInfo.sin_port = htons(1234);
    serverInfo.sin_family = AF_INET;
    serverInfo.sin_addr.s_addr = inet_addr("127.0.0.2");
    serverSize = sizeof(serverInfo);
    
    printf("Number: ");
    if(0 > scanf("%hu", &number))
    {
        printf("error reading number.\n");
        exit(2);
    }

    number = htons(number);
    sendto(clientSocket, &number, sizeof(number), 0, (SOCKADDR*)&serverInfo, serverSize);
    
    recvfrom(clientSocket, &isPrim, sizeof(isPrim), MSG_WAITALL, (SOCKADDR*)&serverInfo, &serverSize);

    isPrim = ntohs(isPrim);
    number = ntohs(number);
    if(isPrim)
        printf("Number %hu is prim.\n", number);
    else
        printf("Number %hu is not prim.\n", number);

    close(clientSocket);
    return 0;
}
