
from xmlrpc.server import SimpleXMLRPCServer
import xmlrpc.client

def sayhello():
    return 'Hello World'

server = SimpleXMLRPCServer(("localhost", 8000))
print("Server is listening")
server.register_function(sayhello, "sayHello")
server.serve_forever()

