#include <iostream>
#include <stdlib.h>
#include <time.h>
#include "Matrix.h"

int main(int argc, char **argv) {

	srand (time(NULL));

	unsigned rows = 3;
	unsigned columns = 4;

	Matrix<int> test(rows, columns, 0.0);

	for ( unsigned i=0; i < rows; i++ ) {
		for (unsigned j=0; j < columns; j++) {
			test(i,j) = rand() % 1000;
		}
	}

	std::cout << test;

	return 0;
}
