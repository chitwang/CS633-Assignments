#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

// Send and Receive buffer arrays
double *sendtop;
double *sendbot;
double *sendleft;
double *sendright;
double *recvtop;
double *recvbot;
double *recvleft;
double *recvright;

// Get the grid point value for row i, column j and grid data
double get_val(double **data, int n, int i, int j) {
    int row,col;
    double val = 0.0;
    if(i >= 0 && i < n && j >= 0 && j < n ) {
       return data[i][j]; 
    } 
    if(i == -1) {
        return recvtop[j];
    }
    if(i == -2) {
        return recvtop[n + j];
    }
    if(i == n) {
        return recvbot[j];
    }
    if(i == n + 1) {
        return recvbot[j + n];
    }
    if(j == -1) {
        return recvleft[i];
    }
    if(j == -2) {
        return recvleft[n + i];
    }
    if(j == n) {
        return recvright[i];
    }
    if(j == n+1) {
        return recvright[i + n];
    }
}

// Starting point of code
int main( int argc, char *argv[]) {
    int rank, size; // Rank of each process and total number of processes

    // Get the values of arguments passed from the command line
    int P_x = atoi(argv[1]);
    int N = atoi(argv[2]);
    int timesteps = atoi(argv[3]);
    int seed = atoi(argv[4]);
    int stencil = atoi(argv[5]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int P_y = size / P_x;
    int n = sqrt(N);

    // Allocating data matrix
    double **data = (double **)malloc(n * sizeof(double *));
    for(int i = 0; i < n; i++){
        data[i] = (double *)malloc(n * sizeof(double));
    }
    
    // Allocating matrix which stores calculated average values
    double **data_avg = (double **)malloc(n * sizeof(double *));
    for(int i = 0; i < n; i++){
        data_avg[i] = (double *)malloc(n * sizeof(double));
    }
    
    // Data Initialisation
    srand(seed * (rank + 10));
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            data[i][j] = abs(rand() + (i * rand() + j * rank)) / 100;
        }
    }

    // Initializing send and receive buffers with 0
    int send_recv_size = (stencil == 5 ? n : n*2);
    int size_pack = send_recv_size * sizeof(double);
    sendtop = (double *)calloc(send_recv_size, sizeof(double));
    sendbot = (double *)calloc(send_recv_size, sizeof(double));
    sendleft = (double *)calloc(send_recv_size, sizeof(double));
    sendright = (double *)calloc(send_recv_size, sizeof(double));
    recvbot = (double *)calloc(send_recv_size, sizeof(double));
    recvtop = (double *)calloc(send_recv_size, sizeof(double));
    recvleft = (double *)calloc(send_recv_size, sizeof(double));
    recvright = (double *)calloc(send_recv_size, sizeof(double));
    double sTime = MPI_Wtime();

    // Loop iterating over time steps
    for(int time_step = 0; time_step < timesteps; time_step++) {        
        if(rank >= P_y) {
            int position = 0;
            for(int j = 0; j < n; j++) {
                MPI_Pack(&data[0][j], 1, MPI_DOUBLE, sendtop, size_pack , &position , MPI_COMM_WORLD);
            }
            if(stencil == 9) {
                for(int j = 0; j < n; j++) {
                    MPI_Pack(&data[1][j], 1, MPI_DOUBLE, sendtop, size_pack , &position , MPI_COMM_WORLD);
                }
            }
            MPI_Status status;
            // First send to upper processes then receive from upper processes
            MPI_Send(sendtop, position, MPI_PACKED, rank - P_y, rank, MPI_COMM_WORLD);
            MPI_Recv(recvtop , send_recv_size, MPI_DOUBLE, rank - P_y, rank - P_y, MPI_COMM_WORLD, &status);
        }
        if(rank + P_y < size) {
            int position = 0;
            for(int j = 0; j< n; j++) {
                MPI_Pack(&data[n - 1][j], 1, MPI_DOUBLE, sendbot, size_pack, &position, MPI_COMM_WORLD);
            }
            if(stencil == 9) {
                for(int j = 0; j < n; j++) {
                    MPI_Pack(&data[n - 2][j], 1, MPI_DOUBLE, sendbot, size_pack , &position , MPI_COMM_WORLD);
                }
            }
            MPI_Status status;
            // First receive from lower processes then send to lower processes
            MPI_Recv(recvbot, send_recv_size, MPI_DOUBLE, rank + P_y , rank + P_y , MPI_COMM_WORLD, &status);
            MPI_Send(sendbot, position, MPI_PACKED, rank + P_y, rank, MPI_COMM_WORLD);
        }
        if(rank % P_y != 0) {
            int position = 0;
            for(int i = 0; i < n; i++) {
                MPI_Pack(&data[i][0], 1, MPI_DOUBLE, sendleft, size_pack, &position, MPI_COMM_WORLD);
            }
            if(stencil == 9) {
                for(int i = 0; i < n; i++) {
                    MPI_Pack(&data[i][1], 1, MPI_DOUBLE, sendleft, size_pack , &position , MPI_COMM_WORLD);
                }
            }
            MPI_Status status;
            // First send to left processes then receive from left processes
            MPI_Send(sendleft, position, MPI_PACKED, rank - 1, rank, MPI_COMM_WORLD);
            MPI_Recv(recvleft , send_recv_size , MPI_DOUBLE, rank - 1 , rank - 1 , MPI_COMM_WORLD, &status);;
        }
        if(rank % P_y != P_y - 1) {
            int position = 0;
            for(int i = 0; i < n; i++) {
                MPI_Pack( &data[i][n - 1] , 1 , MPI_DOUBLE , sendright , size_pack , &position , MPI_COMM_WORLD);
            }
            if(stencil == 9) {
                for(int i = 0; i < n; i++) {
                    MPI_Pack(&data[i][n - 2], 1, MPI_DOUBLE, sendright, size_pack , &position , MPI_COMM_WORLD);
                }
            }
            MPI_Status status;
            // First receive from right processes then send to right processes
            MPI_Recv( recvright, send_recv_size , MPI_DOUBLE, rank + 1, rank + 1, MPI_COMM_WORLD, &status);
            MPI_Send(sendright, position, MPI_PACKED, rank + 1, rank, MPI_COMM_WORLD);
        }
        
        // Iterate over rows and columns to calculate sum and average
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                int count = stencil;
                double sum = data[i][j];

                // Calculate count for each grid point
                if(rank < P_y) {
                    if(i == 0) {
                        count--;
                    }
                    if(stencil == 9) {
                        if(i == 1) {
                            count--;
                        }
                        if(i == 0) {
                            count--;
                        }
                    }
                }
                if(rank + P_y >= size) {
                    if(i == n - 1) {
                        count--;
                    }
                    if(stencil == 9) {
                        if(i == n-2) {
                            count--;
                        }
                        if(i == n-1) {
                            count--;
                        }
                    }
                }
                if(rank % P_y == 0) {
                    if(j == 0) {
                        count--;
                    }
                    if(stencil == 9) {
                        if(j == 1) {
                            count--;
                        }
                        if(j == 0) {
                            count--;
                        }
                    }
                }
                if(rank % P_y == P_y - 1) {
                    if(j == n-1) {
                        count--;
                    }
                    if(stencil == 9) {
                        if(j == n-2) {
                            count--;
                        }
                        if(j == n-1) {
                            count--;
                        }
                    }
                }
                
                sum += get_val(data, n, i - 1, j) + get_val(data, n, i + 1, j) + get_val(data, n, i, j - 1) + get_val(data, n, i, j + 1);
                if(stencil == 9) {
                    sum += get_val(data, n, i - 2, j) + get_val(data, n, i + 2, j) + get_val(data, n, i, j - 2) + get_val(data, n, i, j + 2);
                }
                data_avg[i][j] = sum / count;
            }
        }
        
        // Swap data with data_avg
        double **temp = data;
        data = data_avg;
        data_avg = temp;
    }
    double eTime = MPI_Wtime();
    double local_time = eTime - sTime;
    double max_time;
    // Get maximum of time taken by each process at process 0
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(!rank) {
        printf("%lf\n", max_time);
    }
    MPI_Finalize();
    return 0;
}