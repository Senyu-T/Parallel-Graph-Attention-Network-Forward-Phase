run_omp: run_omp.c
	gcc -Wall -Wextra -Werror -std=c99 -o run_omp gat_omp.c run_omp.c utils_omp.c -lm -fopenmp

clean:
	rm -f run_omp *.o
