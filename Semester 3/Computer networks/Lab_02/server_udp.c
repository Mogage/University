#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct sockaddr_in SOCKADDR_IN;
typedef struct sockaddr SOCKADDR;
#define FALSE 0
#define TRUE 1

int checkPrim(int Number)
{
    if(Number % 2 == 0 && Number > 2)
        return FALSE;
    for(int count = 3; count * count < Number; count = count + 2)
        if(Number % count == 0)
            return FALSE;
    return TRUE;
}

int main()
{
    int serverSocket;
    SOCKADDR_IN serverInfo;
    SOCKADDR_IN clientInfo;
    int clientSize;
    int isPrim;
    uint16_t number;
    
    serverSocket = socket(AF_INET, SOCK_DGRAM, 0);
    if ( serverSocket < 0 )
    {
        printf("error creating socket\n");
        exit(1);
    }

    memset(&serverInfo, 0, sizeof(serverInfo));
    serverInfo.sin_port = htons(1234);
    serverInfo.sin_family = AF_INET;
    serverInfo.sin_addr.s_addr = INADDR_ANY;

    if(bind(serverSocket, (SOCKADDR*)&serverInfo, sizeof(serverInfo)) < 0)
    {
        printf("Error at bind\n");
        exit(2);
    }

    printf("Server up & running.\n");

    clientSize = sizeof(clientInfo);
    memset(&clientInfo, 0, clientSize);

    while(TRUE)
    {

        recvfrom(serverSocket, &number, sizeof(number), MSG_WAITALL, (SOCKADDR*)&clientInfo, &clientSize);
        number = ntohs(number);
        printf("Number received: %hu\n", number);
 
        isPrim = checkPrim(number);
        isPrim = ntohs(isPrim);
        sendto(serverSocket, &isPrim, sizeof(isPrim), 0, (SOCKADDR*)&clientInfo, clientSize);

    }    

    close(serverSocket);
    return 0;
}
