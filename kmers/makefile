O = 3
EIGEN_LIB = ~/github/stan-dev/cmdstan/stan/lib/stan_math/lib/eigen_3.3.9
TRANSCRIPTOME = data/unpacked/GRCh38_latest_rna.fna
BINARY_OUT_FILE = xt.bin
CPP = /usr/local/Cellar/llvm/13.0.0_1/bin/clang++
# CPP = /usr/local/Cellar/gcc/11.2.0_2/bin/g++-11
COMPILE = ${CPP} -O${O} -g -std=c++14 -I src/ -I ${EIGEN_LIB}

builder : src/builder.cpp
	${COMPILE} -o builder src/builder.cpp

xt.bin: builder
	time ./builder ${TRANSCRIPTOME} ${BINARY_OUT_FILE}

reader: src/reader.cpp src/kmers/multinomial-model.hpp
	${COMPILE} -o reader src/reader.cpp

read: xt.bin reader
	time ./reader ${BINARY_OUT_FILE}

clean:
	rm -fR .DS_Store data/.DS_Store builder builder.dSYM reader reader.dSYM

clean-all: clean
	rm -f xt.bin
