#include "Server.h"
#pragma comment (lib, "ws2_32.lib")
#include "WSAInitializer.h"


int main()
{

	try
	{
		WSAInitializer wsaInit;
		Server myServer;
		myServer.run();
	}
	catch (std::exception& e)
	{
		std::cout << "Error occured: " << e.what() << std::endl;
	}


	return 0;
}
