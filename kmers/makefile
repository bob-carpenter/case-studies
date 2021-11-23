O = 3
EIGEN_LIB = ~/github/stan-dev/cmdstan/stan/lib/stan_math/lib/eigen_3.3.9

xtBuilder : src/main.cpp
	clang++ -O${O} -g -std=c++14 -I src/ -I ${EIGEN_LIB} -o xtBuilder src/main.cpp

xt.bin: xtBuilder
	time ./xtBuilder data/unpacked/GRCh38_latest_rna.fna xt.bin

reader: src/reader.cpp
	clang++ -O${O} -g -std=c++14 -I src/ -I ${EIGEN_LIB} -o reader src/reader.cpp

read: xt.bin reader
	time ./reader

clean:
	rm -fR .DS_Store data/.DS_Store xtBuilder xtBuilder.dSYM reader reader.dSYM

clean-all: clean
	rm xt.bin
