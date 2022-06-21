CC = mpicc
CFLAGS = -Wall -lm -fopenmp -g -O2
EXEC = exec
INCLUDE = ./headers/
all : ./par
par: ./par
clean : cleanmaster
./lib/libOperation.a : lib/Lib_basic_fonction_paralel.c headers/Lib_basic_fonction_paralel.h
		$(CC) -o ./lib/Lib_basic_fonction.o -c $< -I$(INCLUDE) $(CFLAGS)
		ar rcs $@ ./lib/Lib_basic_fonction.o

./par: par.c ./lib/libOperation.a
		$(CC) $< -I$(INCLUDE) -L./lib -lOperation -o $@ $(CFLAGS)

clean:
	rm -f *.o
	rm  par
#################
##### CLEAN #####
#################
cleanmaster :
	make cleanlib

cleanlib :
	rm -f ./lib/*.a
	rm -f ./lib/*.so
	rm -f ./lib/*.o
	rm -f 2*.txt



