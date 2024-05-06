#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
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
double **data;
double **data_avg;

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
	if(j == n + 1) {
		return recvright[i + n];
	}
}

void alloc_data(int n) {
	// Allocating data matrix
	data = (double **)malloc(n * sizeof(double *));
	for(int i = 0; i < n; i++) {
		data[i] = (double *)malloc(n * sizeof(double));
	}
	
	// Allocating matrix which stores calculated average values
	data_avg = (double **)malloc(n * sizeof(double *));
	for(int i = 0; i < n; i++) {
		data_avg[i] = (double *)malloc(n * sizeof(double));
	}
	int send_recv_size = n * 2;
	sendtop = (double *)calloc(send_recv_size, sizeof(double));
	sendbot = (double *)calloc(send_recv_size, sizeof(double));
	sendleft = (double *)calloc(send_recv_size, sizeof(double));
	sendright = (double *)calloc(send_recv_size, sizeof(double));
	recvbot = (double *)calloc(send_recv_size, sizeof(double));
	recvtop = (double *)calloc(send_recv_size, sizeof(double));
	recvleft = (double *)calloc(send_recv_size, sizeof(double));
	recvright = (double *)calloc(send_recv_size, sizeof(double));
}

// Data Initialisation
void init_data(int seed, int rank, int n) {
	srand(seed * (rank + 10));
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			data[i][j] = abs(rand() + (i * rand() + j * rank));
		}
	}
}

// free all the buffers which were dynamically allocated
void free_buf(int n) {
	free(sendtop);
	free(sendbot);
	free(sendleft);
	free(sendright);
	free(recvtop);
	free(recvbot);
	free(recvleft);
	free(recvright);
	
	for(int i = 0; i < n; i++) {
		free(data[i]);
		free(data_avg[i]);
	}
	free(data);
	free(data_avg);
}

// re-initialises the send-receive buffers to 0 
void clean_send_buf(int n) {
	memset(sendtop, 0, n);
	memset(sendbot, 0, n);
	memset(sendleft, 0, n);
	memset(sendright, 0, n);
	memset(recvtop, 0, n);
	memset(recvbot, 0, n);
	memset(recvleft, 0, n);
	memset(recvright, 0, n);
}

// Main function which performs stencil computation
double do_halo_exchange(int rank, int n, int timesteps, int P_x, int size, int leader) {
	int color = rank / P_x;
	int send_recv_size = n * 2;
	int size_pack = send_recv_size * sizeof(double);
	MPI_Comm new_comm;
	int newrank;
	if(leader) {
		MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);
		MPI_Comm_rank(new_comm, &newrank);
	}
	double *sendsubtop, *sendsubbot, *recvsubbot, *recvsubtop;
	int size_sub_buf = P_x * send_recv_size;
	if(!newrank && leader) {     // leader
		sendsubtop = (double *)calloc(size_sub_buf, sizeof(double));
		sendsubbot = (double *)calloc(size_sub_buf, sizeof(double));
		recvsubtop = (double *)calloc(size_sub_buf, sizeof(double));
		recvsubbot= (double *)calloc(size_sub_buf, sizeof(double));
	}
	double sTime = MPI_Wtime();
	// Loop iterating over time steps
	for(int time_step = 0; time_step < timesteps; time_step++) {        
		if(rank >= P_x) {
			int position = 0;
			for(int j = 0; j < n; j++) {
				MPI_Pack(&data[0][j], 1, MPI_DOUBLE, sendtop, size_pack , &position , MPI_COMM_WORLD);
			}
			for(int j = 0; j < n; j++) {
				MPI_Pack(&data[1][j], 1, MPI_DOUBLE, sendtop, size_pack , &position , MPI_COMM_WORLD);
			}
			
			MPI_Status status;
			if(leader) {
				MPI_Gather(sendtop, position, MPI_PACKED, sendsubtop, send_recv_size, MPI_DOUBLE, 0, new_comm);
				// First send to upper processes then receive from upper processes
				if(!newrank) {
					MPI_Send(sendsubtop, size_sub_buf, MPI_DOUBLE, rank - P_x, rank, MPI_COMM_WORLD);
					MPI_Recv(recvsubtop, size_sub_buf, MPI_DOUBLE, rank - P_x, rank - P_x, MPI_COMM_WORLD, &status);
				}
				MPI_Scatter(recvsubtop, send_recv_size, MPI_DOUBLE, recvtop, send_recv_size, MPI_DOUBLE, 0, new_comm);
			}
			else {
				MPI_Send(sendtop, position, MPI_PACKED, rank - P_x, rank, MPI_COMM_WORLD);
				MPI_Recv(recvtop , send_recv_size, MPI_DOUBLE, rank - P_x, rank - P_x, MPI_COMM_WORLD, &status);
			}
		}
		if(rank + P_x < size) {
			int position = 0;
			for(int j = 0; j< n; j++) {
				MPI_Pack(&data[n - 1][j], 1, MPI_DOUBLE, sendbot, size_pack, &position, MPI_COMM_WORLD);
			}
			for(int j = 0; j < n; j++) {
				MPI_Pack(&data[n - 2][j], 1, MPI_DOUBLE, sendbot, size_pack , &position , MPI_COMM_WORLD);
			}
			MPI_Status status;
			// First receive from lower processes then send to lower processes
			if(leader){
				MPI_Gather(sendbot, position, MPI_PACKED, sendsubbot, send_recv_size, MPI_DOUBLE, 0, new_comm);            
				if(!newrank) {
					MPI_Recv(recvsubbot, size_sub_buf, MPI_DOUBLE, rank + P_x , rank + P_x , MPI_COMM_WORLD, &status);
					MPI_Send(sendsubbot, size_sub_buf, MPI_DOUBLE, rank + P_x, rank, MPI_COMM_WORLD);
				}
				MPI_Scatter(recvsubbot, send_recv_size, MPI_DOUBLE, recvbot, send_recv_size, MPI_DOUBLE, 0, new_comm);
			}
			else{
				MPI_Recv(recvbot, send_recv_size, MPI_DOUBLE, rank + P_x , rank + P_x , MPI_COMM_WORLD, &status);
				MPI_Send(sendbot, position, MPI_PACKED, rank + P_x, rank, MPI_COMM_WORLD);
			}
		}
		if(rank % P_x != 0) {
			int position = 0;
			for(int i = 0; i < n; i++) {
				MPI_Pack(&data[i][0], 1, MPI_DOUBLE, sendleft, size_pack, &position, MPI_COMM_WORLD);
			}
			for(int i = 0; i < n; i++) {
				MPI_Pack(&data[i][1], 1, MPI_DOUBLE, sendleft, size_pack , &position , MPI_COMM_WORLD);
			}
			MPI_Status status;
			// First send to left processes then receive from left processes
			MPI_Send(sendleft, position, MPI_PACKED, rank - 1, rank, MPI_COMM_WORLD);
			MPI_Recv(recvleft , send_recv_size , MPI_DOUBLE, rank - 1 , rank - 1 , MPI_COMM_WORLD, &status);;
		}
		if(rank % P_x != P_x - 1) {
			int position = 0;
			for(int i = 0; i < n; i++) {
				MPI_Pack(&data[i][n - 1], 1, MPI_DOUBLE, sendright, size_pack, &position, MPI_COMM_WORLD);
			}
			for(int i = 0; i < n; i++) {
				MPI_Pack(&data[i][n - 2], 1, MPI_DOUBLE, sendright, size_pack , &position , MPI_COMM_WORLD);
			}
			MPI_Status status;
			// First receive from right processes then send to right processes
			MPI_Recv(recvright, send_recv_size, MPI_DOUBLE, rank + 1, rank + 1, MPI_COMM_WORLD, &status);
			MPI_Send(sendright, position, MPI_PACKED, rank + 1, rank, MPI_COMM_WORLD);
		}
		// Iterate over rows and columns to calculate sum and average
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < n; j++) {
				int count = 9;
				double sum = data[i][j];
				// Calculate count for each grid point
				if(rank < P_x) {
					if(i == 0) {
						count-=2;
					}
					if(i == 1) {
						count--;
					}
				}
				if(rank + P_x >= size) {
					if(i == n - 1) {
						count-=2;
					}
					if(i == n-2) {
						count--;
					}
				}
				if(rank % P_x == 0) {
					if(j == 0) {
						count-=2;
					}
					if(j == 1) {
						count--;
					}
				}
				if(rank % P_x == P_x - 1) {
					if(j == n-1) {
						count-=2;
					}
					if(j == n-2) {
						count--;
					}
				}
				sum += get_val(data, n, i - 1, j) + get_val(data, n, i + 1, j) + get_val(data, n, i, j - 1) + get_val(data, n, i, j + 1);
				sum += get_val(data, n, i - 2, j) + get_val(data, n, i + 2, j) + get_val(data, n, i, j - 2) + get_val(data, n, i, j + 2);
				data_avg[i][j] = sum / count;
			}
		}
		
		// Swap data with data_avg
		double **temp = data;
		data = data_avg;
		data_avg = temp;
	}
	double eTime = MPI_Wtime();
	if(!newrank && leader) {
		free(sendsubbot);
		free(sendsubtop);
		free(recvsubbot);
		free(recvsubtop);
	}
	double local_time = eTime - sTime;
	return local_time;
}

// Starting point of code
int main(int argc, char *argv[]) {
	int rank, size; // Rank of each process and total number of processes

	// Get the values of arguments passed from the command line
	int P_x = atoi(argv[1]);
	int N = atoi(argv[2]);
	int timesteps = atoi(argv[3]);
	int seed = atoi(argv[4]);
	int stencil = 9;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int P_y = size / P_x;
	int n = sqrt(N);
	
	alloc_data(n);
	
	// Leader can be 0 or 1 depending on the case
	int leader = 0; 
	double local_time[2];
	double max_time[2];
	init_data(seed, rank, n);
	local_time[0] = do_halo_exchange(rank, n, timesteps, P_x, size, leader);
	// if(!rank)
	// 	printf("unaware data %lf\n", data[0][0]);
	clean_send_buf(n * 2);

	leader = 1;
	init_data(seed, rank, n);
	local_time[1] = do_halo_exchange(rank, n, timesteps, P_x, size, leader);
	
	// Get maximum of time taken by each process at process 0
	MPI_Reduce(&local_time, &max_time, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(!rank) {
		printf("Time with leader = %lf\n", max_time[1]);
		printf("Time without leader = %lf\n", max_time[0]);
		printf("Data = %lf\n", data[0][0]);
	}
	
	free_buf(n);
	MPI_Finalize();
	return 0;
}
