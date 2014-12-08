#include <iostream>
#include <fstream>
#include <algorithm>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

using namespace std;

char name[MPI_MAX_PROCESSOR_NAME];

double getClock()
{
    return MPI_Wtime();
    struct timeval tp;
    struct timezone tzp;

    gettimeofday(&tp, &tzp);

    return tp.tv_sec + tp.tv_usec / 1000000.0;
}

void FloydsAlgorithm(int pcount, double *data, int N, int start, int end) {
    int owner = 0;
    for (int k = 0; k < N; ++k) {
        while (k >= N * (owner + 1) / pcount) {
            ++owner;
        }
        MPI_Bcast(&data[k * N], N, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        #pragma omp parallel for collapse(2)
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < N; ++j) {
                data[i * N + j] = min(data[i * N + j], data[i * N + k] + data[k * N + j]);
            }
        }
    }
}

void Server(int pcount, const char *filename) {
    MPI_Status status;

    FILE *I_in;
    // Load in the Adjacency matrix.
    ifstream M_in(filename, ios::in);
    int N, index;
    M_in >> N;

    double *data = new double[N * N];
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            M_in >> data[y * N + x];
        }
    }

    double time = getClock();

    // Broadcast out the matrix width/height.
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Broadcast out the matrix contents.
    MPI_Bcast(data, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    FloydsAlgorithm(pcount, data, N, 0, N / pcount);

    for(int p = 1; p < pcount; ++p) {
        int start = N * p / pcount;
        int end = N * (p + 1) / pcount;
        MPI_Recv(&data[start * N], N * (end - start), MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &status);
    }
    time = getClock() - time;

    // Print the result.
    cout << "tasks: " << pcount << endl;
    cout << "time: " << time << endl;
/*
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << data[i * N + j] << " ";
        }
        cout << endl;
    }
*/
    delete[] data;
}

// Slave process - receives a request, performs Floyd's algorithm, and returns a subset of the data.
void Slave(int pcount, int rank) {
    int N;
    MPI_Status status;

    // Receive broadcast of N (the size of the matrix).
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *data = new double[N * N];

    // Receive the matrix.
    MPI_Bcast(data, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int start = N * rank / pcount;
    int end = N * (rank + 1) / pcount;

    // Run Floyd.
    FloydsAlgorithm(pcount, data, N, start, end);

    // Send my data.
    MPI_Send(&data[start * N], (end - start) * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    delete[] data;
}

int main(int argc, char * argv[]) {
    int pcount, rank, len;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &pcount);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(name, &len);

    const char *filename = argv[1];
#ifndef NO_OMP
    omp_set_num_threads(strtol(argv[2], NULL, 10));
#endif
    if (rank == 0) {
        Server(pcount, filename);
    } else {
        Slave(pcount, rank); 
    }
    MPI_Finalize();
}

