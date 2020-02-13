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



TEST(MatrixTest, ConstructorTest) {

	unsigned rows = 3;
	unsigned cols = 4;
	int initial = 0;
	Matrix<int> A(rows, cols, initial);

	EXPECT_EQ(4, A.getCols());
	EXPECT_EQ(3, A.getRows());
	EXPECT_EQ(12, A.getSize());

	int result;
	for (unsigned i=0; i < rows; i++) {
		for (unsigned j=0; j < cols; j++) {
			result = A(i,j);
			ASSERT_EQ(0, result);
		}
	}
}

TEST(MatrixTest, AccessTest) {

	unsigned rows = 3;
	unsigned cols = 4;
	int initial = 0;
	Matrix<int> A(rows, cols, initial);
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

	Matrix<int> A(rows, cols , initial);

	ASSERT_EQ(rows, A.getRows());
}

TEST(MatrixTest, getColsTest) {
	unsigned rows = 3;
	unsigned cols = 4;
	int initial = 0;

	Matrix<int> A(rows, cols , initial);

	ASSERT_EQ(cols, A.getCols());
}

TEST(MatrixTest, getSizeTest) {
	unsigned rows = 3;
	unsigned cols = 4;
	int initial = 0;
	unsigned size = rows * cols;

	Matrix<int> A(rows, cols , initial);

	ASSERT_EQ(size, A.getSize());
}

TEST(MatrixTest, matrixAdditionTest) {
	Matrix<int> A(3,4,0);
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

	Matrix<int> B(4,3,0);
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

	// Matrix Addition
	Matrix<int> expected(3,4,0);
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
	Matrix<int> Result(3,4,0);
	Result = A + A;
	for (unsigned i=0; i < A.getRows(); i++) {
		for (unsigned j=0; j < A.getCols(); i++) {
			EXPECT_EQ(expected(i,j), Result(i,j));
		}
	}

	// Matrix Addition Cumulative
	Matrix<int> temp = A;
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
	temp += A;
	for (unsigned i=0; i < A.getRows(); i++) {
		for (unsigned j=0; j < A.getCols(); i++) {
			EXPECT_EQ(expected(i,j), temp(i,j));
		}
	}

	// Exception Thrown cases
	ASSERT_THROW((Result = A + B), MatrixException);
	ASSERT_THROW((A += B), MatrixException);
}

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}


