run: run.c
	gcc -Wall -Wextra -Werror -std=c99 -o run gat.c run.c utils.c -lm -fopenmp

clean:
	rm -f run *.o
