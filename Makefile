CC = icc
CFLAGS = -std=c11 -qopenmp -mkl=parallel
OBJS = boundary.o data_handler.o initialization.o lpmc_project.o lpm_basic.o neighbor.o constitutive.o solver.o stiffness.o 

lpm: $(OBJS)
	$(CC) -O3 $(CFLAGS) $(OBJS) -o lpmc

clean: 
	rm -rf *.o result*

run: 
	@$(MAKE) && OMP_NUM_THREADS=20 ./lpmc # optimize the threads number for different systems
