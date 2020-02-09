/*
 * Matrix.h
 *
 *  Created on: Feb 8, 2020
 *      Author: jwredhead
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>
#include <ostream>


template < typename T> class Matrix {
public:

	Matrix(unsigned rows, unsigned columns, const T& initial);	// Constructor for defining Matrix with n rows, m columns
	Matrix(unsigned rows, const T& initial); 						// Constructor for defining square Matrix with n rows, n columns

	virtual ~Matrix(); 							// Destructor
	Matrix& operator=(const Matrix<T> &rhs); 	// Assign Constructor
	Matrix(const Matrix<T> &rhs); 			// Copy Constructor

	// Matrix Operations
	Matrix<T>& operator+(const Matrix<T>& rhs);
	Matrix<T>& operator+=(const Matrix<T>& rhs);
	Matrix<T>& operator-(const Matrix<T>& rhs);
	Matrix<T>& operator-=(const Matrix<T>& rhs);
	Matrix<T>& operator*(const Matrix<T>& rhs);
	Matrix<T>& operator*=(const Matrix<T>& rhs);
	Matrix<T>& hadamardProduct(const Matrix<T>& rhs);
	Matrix<T>& transpose();


	// Scalar Operations
	Matrix<T>& operator+(const T& rhs);
	Matrix<T>& operator-(const T& rhs);
	Matrix<T>& operator*(const T& rhs);
	Matrix<T>& operator/(const T& rhs);

	// Vector Operations
	std::vector<T>& operator*(const std::vector<T>& rhs);
	std::vector<T>& diagVec();

	// Member access operations
	T& operator()(const unsigned& row, const unsigned& col);
	const T& operator()(const unsigned& row, const unsigned& col) const;

	unsigned getRows() const;
	unsigned getCols() const;
	unsigned getSize() const;

private:
	unsigned long long m_size;
	unsigned m_rows;
	unsigned m_cols;
	std::vector< std::vector<T> > mat;

};

// Ostream Operator
template<typename U>
inline std::ostream& operator<<(std::ostream&os, const Matrix<U>& matrix) {

	unsigned rows = matrix.getRows();
	unsigned cols = matrix.getCols();
	unsigned size = matrix.getSize();

	os << "Size: " << size << std::endl
			<< "Rows: "<< rows << std::endl
			<< "Columns: "<< cols << std::endl
			<< "Matrix: " << std::endl;

	for (unsigned i=0; i < rows; i++) {
		for (unsigned j=0; j < cols; j++) {
			os<< "/t" << matrix(i,j);
		}
		os << std::endl;
	}

	return os;
}

#include "Matrix.cpp" // Required for template class, compiler need implementation within same file



#endif /* MATRIX_H_ */
