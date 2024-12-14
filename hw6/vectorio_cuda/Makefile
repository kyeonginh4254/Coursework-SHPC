TARGET=main
OBJECTS=main.o

CXX=/usr/local/cuda/bin/nvcc
CC=/usr/local/cuda/bin/nvcc
CFLAGS= -O3 -gencode arch=compute_75,code=sm_75
LDLIBS=-lm

# cuda settings
CFLAGS+=-I/usr/local/cuda/include
LDLIBS+=-L/usr/local/cuda/lib64

all: $(TARGET)

ptx: $(TARGET)
	$(CXX) $(CFLAGS) -ptx main.cu

profile: $(TARGET)
	ncu -o profile -f --set full ./main

$(TARGET): $(OBJECTS)

%.o: %.cu
	$(CXX) $(CFLAGS) -c $<

clean:
	rm -rf $(TARGET) $(OBJECTS)
	rm -rf *.ptx

