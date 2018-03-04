CC = g++
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lm -lgslcblas
INCLUDES = -I<gsl path> -I<eigen path>
LIBS = -L<gsl path>

mve : model.cpp
	$(CC) $(CFLAGS) model.cpp -o mve $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas

