SHELL = /bin/sh

OS=Linux

# libraries ...
Linux_LIB = -qmkl -qopenmp -lpthread -lfftw3_omp -lfftw3 -lm
Linux_DYLIB = -lmkl_intel_ilp64 -lmkl_lapack95_ilp64 -lmkl_blas95_ilp64 -lmkl_intel_thread -lmkl_core -lm

LIB = $(Linux_DYLIB) $(Linux_LIB)

# flags ...
# -MMD -MP auto-generate per-object header dependency files (*.d): -MMD records
# the project headers each .c includes, -MP emits phony targets for them so a
# deleted/renamed header does not break the build. The .d files are pulled in via
# the -include below, so editing a header (e.g. fd.h) rebuilds exactly the
# objects that include it -- no more blanket `make clean`.
Linux_DEPFLAGS = -MMD -MP
Linux_OPTFLAGS = -DMKL_ILP64 -DFFTW_FFT
Linux_CFLAGS = $(Linux_OPTFLAGS) $(Linux_DEPFLAGS) -O3 -qopenmp -mavx2
#Linux_CFLAGS = -g -qopenmp -O0 -Wall
#Linux_CFLAGS = -g -O0 -Wall # uncomment this line to debug/valgrind

MAINNAM = Filter

# compiler ...
Linux_CC = cc

OBJECTS = \
	main.o mod_init.o mod_mem.o mod_pseudopot.o mod_filter.o mod_ortho.o mod_diag.o mod_sigma.o mod_output.o mod_optional_output.o mod_portho.o mod_spinpol.o\
	read.o init.o save.o projectors.o norm.o rand.o interpolate.o strain.o coeff.o angular.o \
	energy.o hamiltonian.o filter.o ortho.o Hmat.o Hmat_mpi.o write.o nerror.o integral.o phamiltonian.o \
	vector.o aux.o init_periodic.o ghamiltonian.o

CUBEOBJECTS = \
	makecube.o write.o size.o init.o nerror.o read.o save.o interpolate.o rand.o norm.o projectors.o vector.o strain.o aux.o

# compilation ...
# compilation rule for all .c files
.c.o:
	$(Linux_CC) -DOS_$(OS) $(Linux_CFLAGS) -c $*.c

# Pull in the auto-generated header dependencies (one .d per object, written by
# the -MMD -MP flags above). The leading '-' silences the "no such file" notice
# on a fresh tree where no .d files exist yet. After the first build, editing a
# header rebuilds only the objects that include it.
-include $(OBJECTS:.o=.d)
-include $(CUBEOBJECTS:.o=.d)

# Compile the main executable Filter.x and link libraries
$(MAINNAM): $(OBJECTS)
	$(Linux_CC) -o $(MAINNAM)_mpi.x $(OBJECTS) $(LIB)


# If desired, run make cube to build the makecube.x executable
cube: $(CUBEOBJECTS)
	$(Linux_CC) -o makecube.x $(CUBEOBJECTS) $(LIB)

clean:
	/bin/rm -f *.o *.x *.d
