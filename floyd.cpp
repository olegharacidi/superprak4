#include <iostream>
#include <fstream>
#include <algorithm>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>
using namespace std;

// Define a smaller infinity constant
#define SMINF 999999

char name[MPI_MAX_PROCESSOR_NAME];

/**
 * GetClock
 * Used for benchmarking performance
 */
double getClock()
{
    struct timeval tp;
    struct timezone tzp;

    gettimeofday(&tp, &tzp);

    return tp.tv_sec + tp.tv_usec / 1000000.0;
}

void FloydsAlgorithm(int rank, int pcount, int *data, int N) {
	int start = N * rank / pcount;
	int end = N * (rank + 1) / pcount;
	int colbuf[N];
	int owner = 0;
	for (int k = 0; k < N; k++) {
		while (k >= N*(owner+1) / pcount) {
		  owner++;
		}
		MPI_Bcast(&data[k*N], N, MPI_INT, owner, MPI_COMM_WORLD);

		for (int i = 0; i < N; i++) {
			colbuf[i] = data[i*N + k];
		}
		for (int p = 0; p < pcount; p++) {
//			MPI_Bcast(&colbuf[N*p / pcount], N*(p+1)/pcount - N*p/pcount, MPI_INT, p, MPI_COMM_WORLD);
		}
		for (int i = 0; i < N; i++) {
			data[i*N + k] = colbuf[i];
		}

		for (int i = start; i < end; i++) {
			for (int j = 0; j < N; j++) {
				data[i*N + j] = min(data[i*N + j], data[i*N + k] + data[k*N + j]);
			}
		}
	}
}

void Server(int pcount, char * file) {
	MPI_Status status;

	FILE *I_in;
	// Load in the Adjacency matrix to test
	ifstream M_in(file, ios::in);
	int N,tmp,index;
	M_in >> N;

	// Generate the dataset
	int data[N*N];
	for (int y = 0; y < N; y++)
		for (int x = 0; x < N; x++) {
			M_in >> tmp;
			data[y*N + x] = tmp;
		}

	double time = getClock();

	// Broadcast out the matrix width/height
	MPI_Bcast (&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// Broadcast out the matrix contents
	MPI_Bcast (data, N*N, MPI_INT, 0, MPI_COMM_WORLD);

	FloydsAlgorithm(0, pcount, data, N);

	int t[N*N];
	for(int p = 1; p < pcount; p++) {
		MPI_Recv(&t, N*N, MPI_INT, p, 0, MPI_COMM_WORLD, &status);
		for(int v = 0; v < N*N; v++) {
			data[v] = min(data[v], t[v]);
		}
	}
	time = getClock() - time;
	// Finally, print the result
	cout << pcount << " threads" << endl;
	cout << "Time: "<< time << endl;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << data[i*N + j] << " ";
		}
		cout << endl;
	}
}

// Slave process - receives a request, performs floyd's algorithm, and returns a subset of the data
void Slave(int rank, int pcount) {
	int N;
	MPI_Status status;

	// Receive broadcast of N (the width/height of the matrix)
	MPI_Bcast (&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int data[N * N];

	// Receive the matrix
	MPI_Bcast(&data, N*N, MPI_INT, 0, MPI_COMM_WORLD);

	// Perform my transformations
	FloydsAlgorithm(rank, pcount, data, N);

	// Send my data
	MPI_Send(data, N*N, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

/**
 * Main function
 * Initializes the required mpi communication layer
 * and dispatches both the server and the slave processes
 * The server will also act as a slave to ensure that all the processors are
 * busy.
 */
int main(int argc, char * argv[]) {
	int size, rank, len;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(name, &len);

	char * file;

	// Take a filename as a param
	if (argc > 1) {
		 file = argv[1];
	} else {
		cout << "Please supply a filename" << endl;
		MPI_Finalize();
		return 1;
	}

	if (rank == 0) {
		Server(size, file);
	} else {
		Slave(rank, size); 
	}
	MPI_Finalize();
}
