#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <time.h>

__global__ void calculations_inv(float *d_array, float *ans,int *n, long *iter)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	float *d_ptr_array = d_array;

	int i;
	//obliczanie sumy
	//ustawienie wskaznika na odpowiedni wiersz tablicy 

	if(id < *iter)
	{

		float add = 0.0;
		float mult = 1.0;

		d_ptr_array = d_ptr_array + id; // przejscie do wiersza "id"
		for(i = 0 ; i < *n; i++)
		{
			add += *d_ptr_array * *d_ptr_array;
			d_ptr_array++;
		}

		//ustawienie wskaznika ponownie na poczatek tablicy 
		d_ptr_array = d_array + id;
		
		mult = cos(*d_ptr_array);
		d_ptr_array++;

		for(i = 1 ; i < *n; i++)
		{
			mult = mult * cos(*d_ptr_array/(i+1));
			d_ptr_array++;
		}

		ans[id] = 1 / 40 * add + 1 - mult;
	}
}

int main(int argc, char const *argv[])
{
	if(argc == 5)
	{

		clock_t start, end;
		double used_time;

		start = clock();

		int i; 									 // iterator
		int j;								     // iterator

		//zmienne z lini argumentów wywołania programu
		const int n 					= atoi(argv[1]); // wymiar zadania
		const int I 					= atoi(argv[2]); // liczba iteracji - > obliczenia przeprowadzane sa na wartosci I^n
		const double iter 				= pow(I, n);
		const int Blocks 				= atoi(argv[3]); // liczba bloków GPU
		const int Threads 				= atoi(argv[4]); // liczba watków dla jednego bloku GPU

		//const float x_min 				= -20.0; // minimalna warotsc dziedziny zadania
		const float x_max 				= 20.0; // maksymalna wartosc dziedziny zadania
		const float rand_max 			= RAND_MAX / 40.0; //ograniczenie przedzialu losowania zmiennych

		float *h_random_array = (float* ) malloc(sizeof(float) * n * iter); //do operacji na danych stosowana jest tablica jednowymiarowa ze wzgledu na alokacje pamieci w GPU
		float *h_ptr_iterator = h_random_array;

		float *ans = (float* ) malloc(sizeof(float) * iter);

		//losowanie wartosci i umieszczenie ich w tablicy
		for(i = 0 ; i < iter; i++)
		{
			for(j = 0 ; j < n ; j++)
			{
				*h_ptr_iterator = rand() / rand_max - x_max;
				h_ptr_iterator += 1;
			}
		}

		float *d_random_array; //tablica zmiennych wylosowanych w pamieci GPU
		float *d_ans;  //tablica wynikow
		int *d_n; // wymiar 
		long *d_iter; //ilosc iteratcji

		cudaMalloc((void **)&d_random_array, sizeof(float) * n * iter);
		cudaMalloc((void **)&d_ans, sizeof(float) * iter);
		cudaMalloc((void **)&d_n, sizeof(int));
		cudaMalloc((void **)&d_iter, sizeof(long));

		cudaMemcpy(d_random_array, h_random_array, sizeof(float) * n * iter, cudaMemcpyHostToDevice);
		cudaMemcpy(d_ans, ans, sizeof(float) * iter, cudaMemcpyHostToDevice);
		cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_iter, &iter, sizeof(long) , cudaMemcpyHostToDevice);

		printf("Liczba blocków : n [%d] Liczba iteracji [%f] , bloki [%d] watki : [%d]", n , iter, Blocks , Threads);

		calculations_inv<<<Blocks, Threads>>>(d_random_array, d_ans, d_n, d_iter);

		cudaMemcpy(ans, d_ans, sizeof(float) * iter , cudaMemcpyDeviceToHost);

		//szukanie minimum

		float y_min  = ans[0];
		for(i = 0 ; i < iter; i++)
		{
			if(ans[i] < y_min) y_min = ans[i];
		}

		end = clock();
		used_time = ((double) (end - start) / CLOCKS_PER_SEC);

		printf("szukane minimum : %f - czas : %f  \n " , y_min, used_time);

		cudaFree(d_random_array);
		cudaFree(d_ans);
		cudaFree(d_n);
		cudaFree(d_iter);

		free(h_random_array);
		free(ans);
	}
	else
	{
		printf("Invalid program parameters plese type /main2 N I Blocks Threads where \n");
		printf("N - is problem dimension\n");
		printf("I - is number of iteratios\n");
		printf("Blocks - is number of used GPU blocks...max is %d\n", 0);
		printf("Threads- is number of used GPU threads per one block  ... max is %d \n", 0);

		return 0;
	}	

	return 0;
}