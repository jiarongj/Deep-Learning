import xmlrpc.client
proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")
hello = proxy.sayHello()
print("Message from Server %s" %hello)
