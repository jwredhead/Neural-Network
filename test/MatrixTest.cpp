/*
 * Test.cpp
 *
 *  Created on: Feb 11, 2020
 *      Author: jwredhead
 */

#include <iostream>
#include <gtest/gtest.h>
#include <Matrix.h>

class MatrixTest : public testing::Test {
	Matrix<int> *A;

	void setup() {
		A = new Matrix<int>(3,4,0);
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
	}

	void teardown() {
		delete A;
		A = nullptr;
	}
};

TEST_F(MatrixTest, ConstructorTest) {

	EXPECT_EQ(cols, A->getCols());
	EXPECT_EQ(rows, A->getRows());
	EXPECT_EQ(size, A->getSize());

	for (unsigned i=0; i < rows; i++) {
		for (unsigned j=0; j < cols; j++) {
			ASSERT_EQ(0, A(i,j));
		}
	}
}

TEST_F(MatrixTest, AccessTest) {
	EXPECT_EQ(2, A(0,0));
	EXPECT_EQ(5, A(0,1));
	EXPECT_EQ(8, A(0,2));
	EXPECT_EQ(6, A(0,3));
	EXPECT_EQ(2, A(1,0));
	EXPECT_EQ(1, A(1,1));
	EXPECT_EQ(9, A(1,2));
	EXPECT_EQ(5, A(1,3));
	EXPECT_EQ(2, A(2,0));
	EXPECT_EQ(1, A(2,1));
	EXPECT_EQ(3, A(2,2));
	EXPECT_EQ(2, A(2,3));
}

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}


