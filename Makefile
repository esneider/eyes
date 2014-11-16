CPPFLAGS=-std=c++11 -O3 -Wall -Wno-unused-variable

DIRS={.,lib/cvplot,lib/eyeLike}

OPENCVLIB=`pkg-config --libs --cflags opencv`
OPENCVDIR=`pkg-config --variable=prefix opencv`

ARMADILLO=-Ilib/armadillo/include -Llib/armadillo -larmadillo

.PHONY: all setup clean

all:
	g++ ${OPENCVLIB} ${ARMADILLO} ${CPPFLAGS} -I${DIRS} ${DIRS}/*.cpp

setup:
	brew install cmake opencv
	cd lib/armadillo && cmake . && make clean all install DESTDIR=.
	ln -s ${OPENCVDIR} lib/opencv

clean:
	rm -rf a.out*
