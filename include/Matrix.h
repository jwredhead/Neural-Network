/*
 * Matrix.h
 *
 *  Created on: Feb 8, 2020
 *      Author: jwredhead
 */
#pragma once

#include <ostream>
#include <array>


template < typename T> class Matrix {
public:

	Matrix(unsigned rows, unsigned columns);	// Constructor for defining Matrix with n rows, m columns
	Matrix(unsigned rows);						// Constructor for defining square Matrix with n rows, n columns
	Matrix();									// Constructor for Null Matrix
	Matrix(T* arr, unsigned size);				// Construct Matrix from array input

	virtual ~Matrix(); 							// Destructor
	Matrix& operator=(const Matrix<T> &rhs); 	// Assign Constructor
	Matrix(const Matrix<T> &rhs); 			// Copy Constructor

	// Member access operations
	T& operator()(const unsigned& row, const unsigned& col);
	const T& operator()(const unsigned& row, const unsigned& col) const;

	// Matrix Operations
	Matrix<T> operator+(const Matrix<T>& rhs);
	Matrix<T> operator+=(const Matrix<T>& rhs);
	Matrix<T> operator-(const Matrix<T>& rhs);
	Matrix<T> operator-=(const Matrix<T>& rhs);
	Matrix<T> operator*(const Matrix<T>& rhs);
	Matrix<T> operator*=(const Matrix<T>& rhs);
	Matrix<T> hadamardProduct(const Matrix<T>& rhs);
	Matrix<T> transpose();
	Matrix<T> fill(T val);

	// Scalar Operations
	Matrix<T> operator+(const T& rhs);
	Matrix<T> operator-(const T& rhs);
	Matrix<T> operator*(const T& rhs);
	Matrix<T> operator/(const T& rhs);

	unsigned getRows() const;
	unsigned getCols() const;
	unsigned long long getSize() const;

private:
	unsigned long long m_size;
	unsigned m_rows;
	unsigned m_cols;
	T* mat;

};

// Ostream Operator
template<typename U>
inline std::ostream& operator<<(std::ostream& os, const Matrix<U>& matrix) {

	unsigned rows = matrix.getRows();
	unsigned cols = matrix.getCols();
	unsigned size = matrix.getSize();

	os << "Size: " << size << '\n'
			<< "Rows: "<< rows << '\n'
			<< "Columns: "<< cols << '\n'
			<< "Matrix: " << '\n';

	for (unsigned i=0; i < rows; i++) {
		for (unsigned j=0; j < cols; j++) {
			os<< "\t" << matrix(i,j);
		}
		os << '\n';
	}

	return os;
}

