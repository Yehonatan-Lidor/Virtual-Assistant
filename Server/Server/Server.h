#pragma once

#include <WinSock2.h>
#include <Windows.h>
#include <thread>
#include <mutex>
#include <string>
#include "Communicator.h"
#include <algorithm>

using namespace std;

class Server
{
public:
	Server();
	~Server();
	void run();

private:
	Communicator* m_Communicator;
};

