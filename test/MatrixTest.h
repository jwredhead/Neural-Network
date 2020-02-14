/*
 * Test.cpp
 *
 *  Created on: Feb 11, 2020
 *      Author: jwredhead
 */

#include <iostream>
#include <gtest/gtest.h>
#include <Matrix.h>
#include <MatrixException.h>



TEST(MatrixTest, constructorTest) {

	unsigned rows = 3;
	unsigned cols = 4;
	Matrix<int> A(rows, cols);

	EXPECT_EQ(4, A.getCols());
	EXPECT_EQ(3, A.getRows());
	EXPECT_EQ(12, A.getSize());

	int result;
	for (unsigned i=0; i < rows; i++) {
		for (unsigned j=0; j < cols; j++) {
			result = A(i,j);
			EXPECT_EQ(0, result);
		}
	}
}

TEST(MatrixTest, squareConstructorTest) {

	unsigned rows = 3;
	Matrix<int> A(rows);

	EXPECT_EQ(3, A.getCols());
	EXPECT_EQ(3, A.getRows());
	EXPECT_EQ(9, A.getSize());

	int result;
	for (unsigned i=0; i < rows; i++) {
		for (unsigned j=0; j < rows; j++) {
			result = A(i,j);
			EXPECT_EQ(0, result);
		}
	}
}

TEST(MatrixTest, assignConstructorTest) {
	unsigned rows = 3;
	unsigned cols = 4;
	unsigned long long size = rows * cols;
	Matrix<int> A(rows, cols);
	A(0,0) = 2;
	A(0,1) = 5;
	A(0,2) = 8;
	A(0,3) = 6;
	A(1,0) = 2;
	A(1,1) = 1;
	A(1,2) = 9;
	A(1,3) = 5;
	A(2,0) = 2;
	A(2,1) = 1;
	A(2,2) = 3;
	A(2,3) = 2;

	Matrix<int> temp;
	temp= A;

	EXPECT_EQ(rows, temp.getRows());
	EXPECT_EQ(cols, temp.getCols());
	EXPECT_EQ(size, temp.getSize());
	int result = temp(0,0);
	EXPECT_EQ(2, result);
	result = temp(0,1);
	EXPECT_EQ(5, result);
	result = temp(0,2);
	EXPECT_EQ(8, result);
	result = temp(0,3);
	EXPECT_EQ(6, result);
	result = temp(1,0);
	EXPECT_EQ(2, result);
	result = temp(1,1);
	EXPECT_EQ(1, result);
	result = temp(1,2);
	EXPECT_EQ(9, result);
	result = temp(1,3);
	EXPECT_EQ(5, result);
	result = temp(2,0);
	EXPECT_EQ(2, result);
	result = temp(2,1);
	EXPECT_EQ(1, result);
	result = temp(2,2);
	EXPECT_EQ(3, result);
	result = temp(2,3);
	EXPECT_EQ(2, result);

}

TEST(MatrixTest, copyConstructorTest) {
	unsigned rows = 3;
	unsigned cols = 4;
	unsigned long long size = rows * cols;
	Matrix<int> A(rows, cols);
	A(0,0) = 2;
	A(0,1) = 5;
	A(0,2) = 8;
	A(0,3) = 6;
	A(1,0) = 2;
	A(1,1) = 1;
	A(1,2) = 9;
	A(1,3) = 5;
	A(2,0) = 2;
	A(2,1) = 1;
	A(2,2) = 3;
	A(2,3) = 2;

	Matrix<int> temp = A;

	EXPECT_EQ(rows, temp.getRows());
	EXPECT_EQ(cols, temp.getCols());
	EXPECT_EQ(size, temp.getSize());
	int result = temp(0,0);
	EXPECT_EQ(2, result);
	result = temp(0,1);
	EXPECT_EQ(5, result);
	result = temp(0,2);
	EXPECT_EQ(8, result);
	result = temp(0,3);
	EXPECT_EQ(6, result);
	result = temp(1,0);
	EXPECT_EQ(2, result);
	result = temp(1,1);
	EXPECT_EQ(1, result);
	result = temp(1,2);
	EXPECT_EQ(9, result);
	result = temp(1,3);
	EXPECT_EQ(5, result);
	result = temp(2,0);
	EXPECT_EQ(2, result);
	result = temp(2,1);
	EXPECT_EQ(1, result);
	result = temp(2,2);
	EXPECT_EQ(3, result);
	result = temp(2,3);
	EXPECT_EQ(2, result);

}

TEST(MatrixTest, AccessTest) {

	unsigned rows = 3;
	unsigned cols = 4;
	int initial = 0;
	Matrix<int> A(rows, cols);
	A(0,0) = 2;
	A(0,1) = 5;
	A(0,2) = 8;
	A(0,3) = 6;
	A(1,0) = 2;
	A(1,1) = 1;
	A(1,2) = 9;
	A(1,3) = 5;
	A(2,0) = 2;
	A(2,1) = 1;
	A(2,2) = 3;
	A(2,3) = 2;
	int result = A(0,0);
	EXPECT_EQ(2, result);
	result = A(0,1);
	EXPECT_EQ(5, result);
	result = A(0,2);
	EXPECT_EQ(8, result);
	result = A(0,3);
	EXPECT_EQ(6, result);
	result = A(1,0);
	EXPECT_EQ(2, result);
	result = A(1,1);
	EXPECT_EQ(1, result);
	result = A(1,2);
	EXPECT_EQ(9, result);
	result = A(1,3);
	EXPECT_EQ(5, result);
	result = A(2,0);
	EXPECT_EQ(2, result);
	result = A(2,1);
	EXPECT_EQ(1, result);
	result = A(2,2);
	EXPECT_EQ(3, result);
	result = A(2,3);
	EXPECT_EQ(2, result);
}

TEST(MatrixTest, getRowTest) {
	unsigned rows = 3;
	unsigned cols = 4;
	int initial = 0;

	Matrix<int> A(rows, cols);

	ASSERT_EQ(rows, A.getRows());
}

TEST(MatrixTest, getColsTest) {
	unsigned rows = 3;
	unsigned cols = 4;
	int initial = 0;

	Matrix<int> A(rows, cols );

	ASSERT_EQ(cols, A.getCols());
}

TEST(MatrixTest, getSizeTest) {
	unsigned rows = 3;
	unsigned cols = 4;
	int initial = 0;
	unsigned size = rows * cols;

	Matrix<int> A(rows, cols);

	ASSERT_EQ(size, A.getSize());
}

TEST(MatrixTest, matrixAdditionTest) {
	Matrix<int> A(3,4);
	A(0,0) = 2;
	A(0,1) = 5;
	A(0,2) = 8;
	A(0,3) = 6;
	A(1,0) = 2;
	A(1,1) = 1;
	A(1,2) = 9;
	A(1,3) = 5;
	A(2,0) = 2;
	A(2,1) = 1;
	A(2,2) = 3;
	A(2,3) = 2;

	// Matrix Addition
	Matrix<int> expected(3,4);
	expected(0,0) = 4;
	expected(0,1) = 10;
	expected(0,2) = 16;
	expected(0,3) = 12;
	expected(1,0) = 4;
	expected(1,1) = 2;
	expected(1,2) = 18;
	expected(1,3) = 10;
	expected(2,0) = 4;
	expected(2,1) = 2;
	expected(2,2) = 6;
	expected(2,3) = 4;
	Matrix<int> Result = A + A;
	for (unsigned i=0; i < Result.getRows(); i++) {
		for (unsigned j=0; j < Result.getCols(); j++) {
			EXPECT_EQ(expected(i,j), Result(i,j));
		}
	}

	// Matrix Addition Cumulative
	expected(0,0) = 4;
	expected(0,1) = 10;
	expected(0,2) = 16;
	expected(0,3) = 12;
	expected(1,0) = 4;
	expected(1,1) = 2;
	expected(1,2) = 18;
	expected(1,3) = 10;
	expected(2,0) = 4;
	expected(2,1) = 2;
	expected(2,2) = 6;
	expected(2,3) = 4;
	A += A;
	for (unsigned i=0; i < A.getRows(); i++) {
		for (unsigned j=0; j < A.getCols(); j++) {
			EXPECT_EQ(expected(i,j), A(i,j));
		}
	}

	// Exception Thrown cases
	Matrix<int> B(4,3);
	ASSERT_THROW((Result = A + B), MatrixException);
	ASSERT_THROW((A += B), MatrixException);
}

TEST(MatrixTest, matrixSubtractionTest) {
	Matrix<int> A(3,4);
	A(0,0) = 2;
	A(0,1) = 5;
	A(0,2) = 8;
	A(0,3) = 6;
	A(1,0) = 2;
	A(1,1) = 1;
	A(1,2) = 9;
	A(1,3) = 5;
	A(2,0) = 2;
	A(2,1) = 1;
	A(2,2) = 3;
	A(2,3) = 2;



	// Matrix Subtraction
	Matrix<int> expected(3,4);
	expected(0,0) = 0;
	expected(0,1) = 0;
	expected(0,2) = 0;
	expected(0,3) = 0;
	expected(1,0) = 0;
	expected(1,1) = 0;
	expected(1,2) = 0;
	expected(1,3) = 0;
	expected(2,0) = 0;
	expected(2,1) = 0;
	expected(2,2) = 0;
	expected(2,3) = 0;
	Matrix<int> Result = A - A;
	for (unsigned i=0; i < A.getRows(); i++) {
		for (unsigned j=0; j < A.getCols(); j++) {
			EXPECT_EQ(expected(i,j), Result(i,j));
		}
	}

	// Matrix Subtraction
	expected(0,0) = 0;
	expected(0,1) = 0;
	expected(0,2) = 0;
	expected(0,3) = 0;
	expected(1,0) = 0;
	expected(1,1) = 0;
	expected(1,2) = 0;
	expected(1,3) = 0;
	expected(2,0) = 0;
	expected(2,1) = 0;
	expected(2,2) = 0;
	expected(2,3) = 0;
	A -=A;
	for (unsigned i=0; i < A.getRows(); i++) {
		for (unsigned j=0; j < A.getCols(); j++) {
			EXPECT_EQ(expected(i,j), A(i,j));
		}
	}

	// Exception Thrown cases
	Matrix<int> B(4,3);
	EXPECT_THROW((Result = A - B), MatrixException);
	EXPECT_THROW((A -= B), MatrixException);
}

TEST(MatrixTest, matrixMultiplicationTest) {
	Matrix<int> A(3,4);
	A(0,0) = 2;
	A(0,1) = 5;
	A(0,2) = 8;
	A(0,3) = 6;
	A(1,0) = 2;
	A(1,1) = 1;
	A(1,2) = 9;
	A(1,3) = 5;
	A(2,0) = 2;
	A(2,1) = 1;
	A(2,2) = 3;
	A(2,3) = 2;

	Matrix<int> B(4,3);
	B(0,0) = 4;
	B(0,1) = 6;
	B(0,2) = 9;
	B(1,0) = 1;
	B(1,1) = 3;
	B(1,2) = 4;
	B(2,0) = 6;
	B(2,1) = 8;
	B(2,2) = 3;
	B(3,0) = 5;
	B(3,1) = 8;
	B(3,2) = 4;

	// Matrix Multiplication
	Matrix<int> expected(3,3);
	expected(0,0) = 91;
	expected(0,1) = 139;
	expected(0,2) = 86;
	expected(1,0) = 88;
	expected(1,1) = 127;
	expected(1,2) = 69;
	expected(2,0) = 37;
	expected(2,1) = 55;
	expected(2,2) = 39;

	Matrix<int> Result = A * B;

	EXPECT_EQ(3, Result.getRows());
	EXPECT_EQ(3, Result.getCols());
	EXPECT_EQ(9, Result.getSize());
	for (unsigned i=0; i < Result.getRows(); i++) {
		for (unsigned j=0; j < Result.getCols(); j++) {
			EXPECT_EQ(expected(i,j), Result(i,j));
		}
	}


	// Matrix Multiplication
	Matrix<int> temp = A;
	expected(0,0) = 91;
	expected(0,1) = 139;
	expected(0,2) = 86;
	expected(1,0) = 88;
	expected(1,1) = 127;
	expected(1,2) = 69;
	expected(2,0) = 37;
	expected(2,1) = 55;
	expected(2,2) = 39;
	temp *=  B;
	for (unsigned i=0; i < temp.getRows(); i++) {
		for (unsigned j=0; j < temp.getCols(); j++) {
			EXPECT_EQ(expected(i,j), temp(i,j));
		}
	}

	// Exception Thrown cases
	EXPECT_THROW((Result = A * A), MatrixException);
	EXPECT_THROW((A *= A), MatrixException);
}

TEST(MatrixTest, hadamardProductTest) {
	Matrix<int> A(3,4);
	A(0,0) = 2;
	A(0,1) = 5;
	A(0,2) = 8;
	A(0,3) = 6;
	A(1,0) = 2;
	A(1,1) = 1;
	A(1,2) = 9;
	A(1,3) = 5;
	A(2,0) = 2;
	A(2,1) = 1;
	A(2,2) = 3;
	A(2,3) = 2;

	Matrix<int> expected(3,4);
	expected(0,0) = 4;
	expected(0,1) = 25;
	expected(0,2) = 64;
	expected(0,3) = 36;
	expected(1,0) = 4;
	expected(1,1) = 1;
	expected(1,2) = 81;
	expected(1,3) = 25;
	expected(2,0) = 4;
	expected(2,1) = 1;
	expected(2,2) = 9;
	expected(2,3) = 4;
	Matrix<int> Result = A.hadamardProduct(A);
	for (unsigned i=0; i < A.getRows(); i++) {
		for (unsigned j=0; j < A.getCols(); j++) {
			EXPECT_EQ(expected(i,j), Result(i,j));
		}
	}

	// Exception Thrown case
	Matrix<int> B(4,3);
	EXPECT_THROW((A.hadamardProduct(B)), MatrixException);
}

TEST(MatrixTest, transposeTest) {
	Matrix<int> A(3,4);
	A(0,0) = 2;
	A(0,1) = 5;
	A(0,2) = 8;
	A(0,3) = 6;
	A(1,0) = 2;
	A(1,1) = 1;
	A(1,2) = 9;
	A(1,3) = 5;
	A(2,0) = 2;
	A(2,1) = 1;
	A(2,2) = 3;
	A(2,3) = 2;

	Matrix<int> expected(4,3);
	expected(0,0) = 2;
	expected(1,0) = 5;
	expected(2,0) = 8;
	expected(3,0) = 6;
	expected(0,1) = 2;
	expected(1,1) = 1;
	expected(2,1) = 9;
	expected(3,1) = 5;
	expected(0,2) = 2;
	expected(1,2) = 1;
	expected(2,2) = 3;
	expected(3,2) = 2;
	Matrix<int> Result = A.transpose();
	for (unsigned i=0; i < Result.getRows(); i++) {
		for (unsigned j=0; j < Result.getCols(); j++) {
			EXPECT_EQ(expected(i,j), Result(i,j));
		}
	}
}

TEST(MatrixTest, fillTest) {
	Matrix<int> A(3,4);

	A.fill(5);

	for (unsigned i=0; i < A.getRows(); i++) {
		for (unsigned j=0; j < A.getCols(); j++) {
			EXPECT_EQ(5, A(i,j));
		}
	}
}

TEST(MatrixTest, scalarAdditionTest) {
	Matrix<int> A(3,4);
	A(0,0) = 2;
	A(0,1) = 5;
	A(0,2) = 8;
	A(0,3) = 6;
	A(1,0) = 2;
	A(1,1) = 1;
	A(1,2) = 9;
	A(1,3) = 5;
	A(2,0) = 2;
	A(2,1) = 1;
	A(2,2) = 3;
	A(2,3) = 2;

	// Scalar Addition
	Matrix<int> expected(3,4);
	expected(0,0) = 7;
	expected(0,1) = 10;
	expected(0,2) = 13;
	expected(0,3) = 11;
	expected(1,0) = 7;
	expected(1,1) = 6;
	expected(1,2) = 14;
	expected(1,3) = 10;
	expected(2,0) = 7;
	expected(2,1) = 6;
	expected(2,2) = 8;
	expected(2,3) = 7;
	Matrix<int> Result = A + 5;
	for (unsigned i=0; i < A.getRows(); i++) {
		for (unsigned j=0; j < A.getCols(); j++) {
			EXPECT_EQ(expected(i,j), Result(i,j));
		}
	}
}

TEST(MatrixTest, scalarSubtractionTest) {
	Matrix<int> A(3,4);
	A(0,0) = 2;
	A(0,1) = 5;
	A(0,2) = 8;
	A(0,3) = 6;
	A(1,0) = 2;
	A(1,1) = 1;
	A(1,2) = 9;
	A(1,3) = 5;
	A(2,0) = 2;
	A(2,1) = 1;
	A(2,2) = 3;
	A(2,3) = 2;

	// Scalar Subtraction
	Matrix<int> expected(3,4);
	expected(0,0) = -3;
	expected(0,1) = 0;
	expected(0,2) = 3;
	expected(0,3) = 1;
	expected(1,0) = -3;
	expected(1,1) = -4;
	expected(1,2) = 4;
	expected(1,3) = 0;
	expected(2,0) = -3;
	expected(2,1) = -4;
	expected(2,2) = -2;
	expected(2,3) = -3;
	Matrix<int> Result = A - 5;
	for (unsigned i=0; i < A.getRows(); i++) {
		for (unsigned j=0; j < A.getCols(); j++) {
			EXPECT_EQ(expected(i,j), Result(i,j));
		}
	}
}

TEST(MatrixTest, scalarMultiplicationTest) {
	Matrix<int> A(3,4);
	A(0,0) = 2;
	A(0,1) = 5;
	A(0,2) = 8;
	A(0,3) = 6;
	A(1,0) = 2;
	A(1,1) = 1;
	A(1,2) = 9;
	A(1,3) = 5;
	A(2,0) = 2;
	A(2,1) = 1;
	A(2,2) = 3;
	A(2,3) = 2;

	// Scalar Multiplication
	Matrix<int> expected(3,4);
	expected(0,0) = 10;
	expected(0,1) = 25;
	expected(0,2) = 40;
	expected(0,3) = 30;
	expected(1,0) = 10;
	expected(1,1) = 5;
	expected(1,2) = 45;
	expected(1,3) = 25;
	expected(2,0) = 10;
	expected(2,1) = 5;
	expected(2,2) = 15;
	expected(2,3) = 10;
	Matrix<int> Result = A * 5;
	for (unsigned i=0; i < A.getRows(); i++) {
		for (unsigned j=0; j < A.getCols(); j++) {
			EXPECT_EQ(expected(i,j), Result(i,j));
		}
	}
}

TEST(MatrixTest, scalarDivisionTest) {
	Matrix<int> A(3,4);
	A(0,0) = 10;
	A(0,1) = 25;
	A(0,2) = 40;
	A(0,3) = 30;
	A(1,0) = 10;
	A(1,1) = 5;
	A(1,2) = 45;
	A(1,3) = 25;
	A(2,0) = 10;
	A(2,1) = 5;
	A(2,2) = 15;
	A(2,3) = 10;

	// Scalar Division
	Matrix<int> expected(3,4);
	expected(0,0) = 2;
	expected(0,1) = 5;
	expected(0,2) = 8;
	expected(0,3) = 6;
	expected(1,0) = 2;
	expected(1,1) = 1;
	expected(1,2) = 9;
	expected(1,3) = 5;
	expected(2,0) = 2;
	expected(2,1) = 1;
	expected(2,2) = 3;
	expected(2,3) = 2;
	Matrix<int> Result = A / 5;
	for (unsigned i=0; i < A.getRows(); i++) {
		for (unsigned j=0; j < A.getCols(); j++) {
			EXPECT_EQ(expected(i,j), Result(i,j));
		}
	}
}




