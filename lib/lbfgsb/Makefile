# Linux settings.
MEX         = /export/share/apps/MATLAB-R2010b/bin/mex
MATLAB_HOME = /export/share/apps/MATLAB-R2010b
CXX         = g++
F77         = gfortran
CFLAGS      = -O3 -fPIC -pthread 
FFLAGS      = -O3 -fPIC -fexceptions 
LDFORTRAN   = 


TARGET = lbfgsb

OBJS   = solver.o matlabexception.o matlabscalar.o matlabstring.o   \
         matlabmatrix.o arrayofmatrices.o program.o matlabprogram.o \
         lbfgsb.o
         
CFLAGS += -Wall -ansi -DMATLAB_MEXFILE

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CFLAGS) -I$(MATLAB_HOME)/extern/include -o $@ -c $^

%.o: %.f
	$(F77) $(FFLAGS) -o $@ -c $^

$(TARGET): $(OBJS)
	$(MEX) -cxx CXX=$(CXX) CC=$(CXX) FC=$(FCC) LD=$(CXX) $(LDFORTRAN) -lgfortran -lm \
        -O -output $@ $^

clean:
	rm -f *.o $(TARGET).mex*

