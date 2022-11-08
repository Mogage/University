from base64 import decode
import socket

HOST = "193.231.20.20"
PORT = 1234

message = input("String: ").encode()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as clientSocket:
    clientSocket.connect((HOST, PORT))
    clientSocket.send(message)
    data = clientSocket.recv(1024)

    print(f"Number of spaces = {int.from_bytes(data, 'big')}")