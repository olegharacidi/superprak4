#include <iostream>
#include <fstream>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;

void readSubmatrix(double *buf, int M, int N, istream& in) {
    for (int i = 0; i < M * N; i++) {
        in >> buf[i];
    }
}

void printSubmatrix(double *buf, int M, int N, ostream& out) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out << buf[i * N + j] << " ";
        }
        out << endl;
    }
}

void floydsAlgorithm(int pcount, double *data, int M, int N, int rank) {
    int owner = 0;
    int start = N * rank / pcount;
    double *kRowBuf = new double[N];
    double *kRowPtr = NULL;
    for (int k = 0; k < N; k++) {
        while (k >= N * (owner + 1) / pcount) {
            owner++;
        }
        if (owner == rank) {
            kRowPtr = &data[(k - start) * N];
        } else {
            kRowPtr = kRowBuf;
        }
        MPI_Bcast(kRowPtr, N, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        #pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; j++) {
                data[i * N + j] = min(data[i * N + j], data[i * N + k] + kRowPtr[j]);
            }
        }
    }
    delete[] kRowBuf;
}

/*
  Server process - loads the data, sends patches to the slaves, runs Floyd's
  algorithm on its own patch, receives the resulting patches from the slaves
  and prints the result to the output.
*/
void server(int pcount, const char *filename) {
    MPI_Status status;

    ifstream fin(filename, ios::in);
    int N;
    fin >> N;

    // Submatrix for this process.

    double *data = new double[N * (N / pcount)];
    readSubmatrix(data, N / pcount, N, fin);

    // Buffer to send to the other processes.
    double *buf = new double[N * (N / pcount + 1)];

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    for (int p = 1; p < pcount; p++) {
        // Load the submatrix for the process p and sent it out.
        int start = N * p / pcount;
        int end = N * (p + 1) / pcount;
        readSubmatrix(buf, end - start, N, fin);
        MPI_Send(buf, N * (end - start), MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
    }
    fin.close();

    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();

    floydsAlgorithm(pcount, data, N / pcount, N, 0);

    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime() - time;

    // Print the result.
    cout << "time: " << time << endl;
//    printSubmatrix(data, N / pcount, N, cout);
    for (int p = 1; p < pcount; p++) {
        // Reveive submatrix from the process p and print it to the output.
        int start = N * p / pcount;
        int end = N * (p + 1) / pcount;
        MPI_Recv(buf, N * (end - start), MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &status);
        // printSubmatrix(buf, end - start, N, cout);
    }

    delete[] buf;
    delete[] data;
}

/*
  Slave process - receives a request, performs Floyd's algorithm,
  and returns the result.
*/
void slave(int pcount, int rank) {
    int N;
    MPI_Status status;

    // Receive broadcast of N (the size of the matrix).
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int start = N * rank / pcount;
    int end = N * (rank + 1) / pcount;

    double *data = new double[N * (end - start)];

    // Receive the submatrix.
    MPI_Recv(data, N * (end - start), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

    MPI_Barrier(MPI_COMM_WORLD);
    // Run.
    floydsAlgorithm(pcount, data, end - start, N, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    // Send my data.
    MPI_Send(data, N * (end - start), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    delete[] data;
}

int main(int argc, char * argv[]) {
    int pcount, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &pcount);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const char *filename = argv[1];

#ifndef NO_OMP
    int num_threads = strtol(argv[2], NULL, 10);
    omp_set_num_threads(num_threads);
#endif
    if (rank == 0) {
        cout << "tasks: " << pcount << endl;
#ifndef NO_OMP
        cout << "threads per task: " << num_threads << endl;
#endif    
        server(pcount, filename);
    } else {
        slave(pcount, rank); 
    }
    MPI_Finalize();
}
