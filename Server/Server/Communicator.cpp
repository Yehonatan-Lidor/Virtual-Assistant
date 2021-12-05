#include "Communicator.h"


Communicator::Communicator(): m_serverSocket(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP))
{

	if (this->m_serverSocket == INVALID_SOCKET)
		throw exception(__FUNCTION__ " - socket");
}

Communicator::~Communicator()
{

	try
	{

		closesocket(this->m_serverSocket);
	}
	catch (...) {}
}

/// <summary>
/// Method binds the socket to a port and makes it listen for new connections
/// </summary>
void Communicator::bindAndListen()
{
	struct sockaddr_in sa = { 0 };

	sa.sin_port = htons(PORT); // port that server will listen on
	sa.sin_family = AF_INET;
	sa.sin_addr.s_addr = INADDR_ANY;

	if (::bind(this->m_serverSocket, (struct sockaddr*)&sa, sizeof(sa)) == SOCKET_ERROR)
		throw exception(__FUNCTION__ " - bind");

	// Start listening for incoming requests of clients
	if (listen(this->m_serverSocket, SOMAXCONN) == SOCKET_ERROR)
		throw exception(__FUNCTION__ " - listen");
	this->m_output_mutex.lock();
	cout << "Listenning...\n";
	this->m_output_mutex.unlock();
}

/// <summary>
/// Method is responsible for accepting new clients
/// </summary>
void Communicator::startHandleRequests()
{
	this->bindAndListen();
	thread t_queue(&Communicator::handleQueue, this);
	t_queue.detach();
	while (true)
	{
		this->m_output_mutex.lock();
		cout << "Accepting client...\n";
		this->m_output_mutex.unlock();
		// creates new socket for communication with the client
		SOCKET client_socket = ::accept(this->m_serverSocket, NULL, NULL);

		if (client_socket == INVALID_SOCKET)
			throw exception(__FUNCTION__);

		// creates thread for client handlement
		thread t(&Communicator::handleNewClient, this, client_socket);
		t.detach();

		// Start listening for incoming requests of clients
		if (listen(this->m_serverSocket, SOMAXCONN) == SOCKET_ERROR)
			throw exception(__FUNCTION__ " - listen");
		this->m_output_mutex.lock();
		cout << "Listenning...\n";
		this->m_output_mutex.unlock();
	}

}

/// <summary>
/// Method handles a new client
/// </summary>
/// <param name="clientSocket"> (SOCKET) - socket dedicated to the new client</param>
void Communicator::handleNewClient(SOCKET clientSocket)
{
	char buff[MSG_LEN] = { NULL };
	while (true)
	{
		recv(clientSocket, buff, MSG_LEN, NULL);
		this->m_queue_mutex.lock();
		this->m_requests_queue.push(pair<string* ,SOCKET>(new string(buff), clientSocket) );
		cout << this->m_requests_queue.empty() << "\n";
		this->m_queue_mutex.unlock();
		//std::unique_lock<mutex> lock(this->m_queue_mutex);
		//lock.unlock();
		//this->m_cv_queue.notify_one(); // wakes up the message queue handler
		
	}
}

void Communicator::handleQueue()
{

	pair<string*, SOCKET> p;
	while(true)
	{
		//unique_lock<mutex> lock(this->m_queue_mutex);
		//this->m_cv_queue.wait(lock); // wait for client to add new message to the queue
		this->m_queue_mutex.lock(); // lock access to the queue untill the queue is empty

		while (!this->m_requests_queue.empty())
		{
			p = this->m_requests_queue.front(); // get request
			this->m_requests_queue.pop();

			send(p.second, "Basic Response", 15, NULL);

			delete p.first;
		}
		this->m_queue_mutex.unlock();
	}
}
