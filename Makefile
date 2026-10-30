all: hough

hough:	hough.cu pgm.o
	nvcc hough.cu pgm.o -o hough

pgm.o:	common/pgm.cpp common/pgm.h
	g++ -c common/pgm.cpp -o ./pgm.o

run: hough
	./hough runway.pgm

clean:
	rm -f hough pgm.o

.PHONY: all run clean
