#include "Server.h"

Server::Server()
{
	this->m_Communicator = new Communicator();

}

Server::~Server()
{
	delete this->m_Communicator;
}

void Server::run()
{
	cout << "Enter the word 'exit' to end the program\n";

	try
	{
		thread t_connector(&Communicator::startHandleRequests, this->m_Communicator);
		t_connector.detach();
		string command = "";
		while (command != "exit")
		{
			getline(cin, command);
			for_each(command.begin(), command.end(), [](char& c) {
				c = ::tolower(c);
				});
		}
	}
	catch (...)
	{
	}
}
