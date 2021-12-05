
#pragma once

#include <WinSock2.h>
#include <Windows.h>
#include <thread>
#include <mutex>
#include <map>
#include <exception>
#include <iostream>
#include <queue>
#include <typeinfo>

#define PORT 8686 // Listenning port 
#define MSG_LEN 1024 // Maximum Length of a message

using namespace std;

class Communicator
{

public:

	Communicator();
	~Communicator();
	void startHandleRequests();

private:

	// private fields
	SOCKET m_serverSocket; // the socket the server will use for listenning for new clients
	mutex m_output_mutex; // mutex for output of the program
	mutex m_queue_mutex; // mutex for condtion variable field
	condition_variable m_cv_queue; // condition variable for message queue
	queue< pair<string*, SOCKET>> m_requests_queue; // queue contatining all requests the server recives

	// private methods
	void bindAndListen();
	void handleNewClient(SOCKET clientSocket);
	void handleQueue();

};

