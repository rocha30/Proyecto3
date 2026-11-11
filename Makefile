all: hough

hough:	hough.cu pgm.o lineas.o
	nvcc hough.cu pgm.o lineas.o -o hough

pgm.o:	common/pgm.cpp common/pgm.h
	g++ -c common/pgm.cpp -o ./pgm.o

lineas.o:	common/lineas.cpp common/lineas.h common/stb_image_write.h
	g++ -c common/lineas.cpp -o ./lineas.o

run: hough
	./hough runway.pgm

clean:
	rm -f hough pgm.o lineas.o

.PHONY: all run clean
