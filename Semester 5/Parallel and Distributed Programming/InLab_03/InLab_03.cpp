#include <iostream>
#include <mpi.h> 
#include <stdlib.h> 
#include <string>

void printVector(int* vector, int size)
{
	for (int i = 0; i < size; ++i)
	{
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}


int main(int argc, char* argv[])
{
	int myid, numprocs, namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	const int n = 10;
	int a[n], b[n], c[n];

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Get_processor_name(processor_name, &namelen);
	if (myid == 0)
	{
		for (int i = 0; i < n; ++i)
		{
			a[i] = rand() % 10;
			b[i] = rand() % 10;
		}
	}

	int* auxA = new int[n / numprocs];
	int* auxB = new int[n / numprocs];
	int* auxC = new int[n / numprocs];

	MPI_Scatter(a, n / numprocs, MPI_INT, auxA, n / numprocs, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(b, n / numprocs, MPI_INT, auxB, n / numprocs, MPI_INT, 0, MPI_COMM_WORLD);

	for (int i = 0; i < n / numprocs; ++i)
	{
		auxC[i] = auxA[i] + auxB[i];
	}

	MPI_Gather(auxC, n / numprocs, MPI_INT, c, n / numprocs, MPI_INT, 0, MPI_COMM_WORLD);

	if (myid == 0)
	{
		for (int i = n - n % numprocs; i < n; ++i)
		{
			c[i] = a[i] + b[i];
		}

		printVector(a, n);
		printVector(b, n);
		printVector(c, n);
	}

	MPI_Finalize();
}

//void send_recv()
//{
//	int myid, numprocs, namelen;
//	char processor_name[MPI_MAX_PROCESSOR_NAME];
//
//	const int n = 10;
//	int a[n], b[n], c[n];
//
//	MPI_Init(NULL, NULL);
//	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
//	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
//	MPI_Get_processor_name(processor_name, &namelen);
//	if (myid == 0)
//	{
//		for (int i = 0; i < n; ++i)
//		{
//			a[i] = rand() % 10;
//			b[i] = rand() % 10;
//		}
//		int start = 0;
//		int end;
//		int bufferSize = n / (numprocs - 1);
//		int bufferRemainder = n % (numprocs - 1);
//
//		for (int i = 1; i < numprocs; ++i)
//		{
//			end = start + bufferSize;
//			if (bufferRemainder > 0)
//			{
//				end++;
//				bufferRemainder--;
//			}
//			MPI_Send(&start, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
//			MPI_Send(&end, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
//			MPI_Send(a + start, end - start, MPI_INT, i, 0, MPI_COMM_WORLD);
//			MPI_Send(b + start, end - start, MPI_INT, i, 0, MPI_COMM_WORLD);
//			start = end;
//		}
//
//		for (int i = 1; i < numprocs; ++i)
//		{
//			MPI_Recv(&start, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//			MPI_Recv(&end, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//			MPI_Recv(c + start, end - start, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//		}
//
//		printVector(a, n);
//		printVector(b, n);
//		printVector(c, n);
//	}
//	else
//	{
//		int start;
//		int end;
//		MPI_Recv(&start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//		MPI_Recv(&end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//		MPI_Recv(a + start, end - start, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//		MPI_Recv(b + start, end - start, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//		printf("process %d: start = %d, end = %d\n", myid, start, end);
//
//		for (int i = start; i < end; ++i)
//		{
//			c[i] = a[i] + b[i];
//		}
//
//		MPI_Send(&start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
//		MPI_Send(&end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
//		MPI_Send(c + start, end - start, MPI_INT, 0, 0, MPI_COMM_WORLD);
//	}
//	MPI_Finalize();
//
//}