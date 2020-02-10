/*
 * Matrix.cpp
 *
 *  Created on: Feb 8, 2020
 *      Author: jwredhead
 */

#include "Matrix.h"
#include "MatrixException.h"



// NxM Constructor
template<typename T>
Matrix<T>::Matrix(unsigned rows, unsigned columns, const T &initial) :
		m_rows(rows), m_cols(columns) {
	m_size = rows * columns;
	mat.resize(rows);
	for (unsigned i=0; i < rows; i++) {
		mat[i].resize(columns, initial);
	}
}

// NxN Constructor
template<typename T>
Matrix<T>::Matrix(unsigned rows, const T &initial) :
		m_rows(rows), m_cols(rows) {
	m_size = rows * rows;
	mat.resize(rows);
	for (unsigned i=0; i< rows; i++) {
		mat[i].resize(rows, initial);
	}
}

// Virtual Destructor, set to default
template<typename T>
Matrix<T>::~Matrix() = default;

// Assign Constructor
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T> &rhs) {
	if (&rhs == this)
		return *this;

	unsigned newRows = rhs.getRows();
	unsigned newCols = rhs.getCols();
	unsigned newSize = rhs.getSize();

	mat.resize(newRows);
	for (unsigned i=0; i < newRows; i++)  {
		mat[i].resize(newCols);
	}

	for (unsigned i=0; i < newRows; i++) {
		for (unsigned j=0; j < newCols; j++) {
			mat[i][j] = rhs(i, j);
		}
	}
	m_rows = newRows;
	m_cols = newCols;
	m_size = newSize;

	return *this;
}

// Copy Constructor
template<typename T>
Matrix<T>::Matrix(const Matrix<T> &rhs) {
	mat = rhs.mat;
	m_rows = rhs.getRows();
	m_cols = rhs.getCols();
	m_size = rhs.getSize();
}

// Addition Operator
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &rhs) {
	if (this->m_rows != rhs.getRows() || this->m_cols != rhs.getCols()) {
		throw MatrixException(__FILE__, __LINE__, __PRETTY_FUNCTION__, "Error: Matrix rows and columns must be equal for matrix addition");
	}

	Matrix<T> result = Matrix(this->m_rows, this->m_cols, 0.0);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; this->m_cols; j++) {
			result(i,j) = this->mat[i][j] + rhs(i,j);
		}
	}

	return result;
}


// Cumulative Addition Operator
template<typename T>
Matrix<T> Matrix<T>::operator+=(const Matrix<T> &rhs) {
	if (this->m_rows != rhs.getRows() || this->m_cols != rhs.getCols()) {
		throw MatrixException(__FILE__, __LINE__, __PRETTY_FUNCTION__, "Error: Matrix rows and columns must be equal for matrix addition");
	}

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; this->m_cols; j++) {
			this->mat[i][j] += rhs(i,j);
		}
	}

	return *this;
}

// Subtraction Operator
template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &rhs) {
	if (this->m_rows != rhs.getRows() || this->m_cols != rhs.getCols()) {
		throw MatrixException(__FILE__, __LINE__, __PRETTY_FUNCTION__, "Error: Matrix rows and columns must be equal for matrix subtraction");
	}

	Matrix<T> result = Matrix(this->m_rows, this->m_cols, 0.0);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; this->m_cols; j++) {
			result(i,j) = this->mat[i][j] - rhs(i,j);
		}
	}

	return result;
}


// Cumulative Subtraction Operator
template<typename T>
Matrix<T> Matrix<T>::operator-=(const Matrix<T> &rhs) {
	if (this->m_rows != rhs.getRows() || this->m_cols != rhs.getCols()) {
		throw MatrixException(__FILE__, __LINE__, __PRETTY_FUNCTION__, "Error: Matrix rows and columns must be equal for matrix subtraction");
	}

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; this->m_cols; j++) {
			this->mat[i][j] -= rhs(i,j);
		}
	}

	return *this;
}

// Multiplication Operator
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &rhs) {
	if (this->m_cols != rhs.getRows()) {
		throw MatrixException(__FILE__, __LINE__, __PRETTY_FUNCTION__, "Error: Rows of Matrix A and columns of Matrix B must be equal for matrix multiplication");
	}

	Matrix<T> result = Matrix(this->m_rows, rhs.getCols(), 0.0);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; j < rhs.getCols(); j++) {
			for (unsigned k ; k < this->m_cols; k++) {
				result(i,j) += this->mat[i][k] * rhs(k,j);
			}
		}
	}

	return result;
}

// Cumulative Multiplication Operator
template<typename T>
Matrix<T> Matrix<T>::operator*=(const Matrix<T> &rhs) {
	if (this->m_cols != rhs.getRows()) {
		throw MatrixException(__FILE__, __LINE__, __PRETTY_FUNCTION__, "Error: Matrix rows and columns must be equal for matrix multiplication");
	}

	Matrix<T> result = Matrix(this->m_rows, rhs.getCols(), 0.0);

	result = (*this) * rhs;

	(*this) = result;

	return *this;
}

// Hardamard Product
template<typename T>
Matrix<T> Matrix<T>::hadamardProduct(const Matrix<T> &rhs) {
	if (this->m_cols != rhs.getRows()) {
		throw MatrixException(__FILE__, __LINE__, __PRETTY_FUNCTION__, "Error: Matrix rows and columns must be equal for Hardamard Product");
	}

	Matrix<T> result = Matrix(this->m_rows, this->m_cols);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; this->m_cols; j++) {
			result(i,j) = this->mat[i][j] * rhs(i,j);
		}
	}

	return result;
}

// Transpose
template<typename T>
Matrix<T> Matrix<T>::transpose() {
	Matrix<T> result = Matrix(this->m_cols, this->m_rows, 0.0);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; this->m_cols; j++) {
			result(i,j) =this->mat[j][i];
		}
	}

	return result;
}

// Scalar Addition
template<typename T>
Matrix<T> Matrix<T>::operator+(const T& rhs) {
	Matrix<T> result = Matrix(this->m_rows, this->m_cols, 0.0);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; this->m_cols; j++) {
			result(i,j) = this->mat[i][j] + rhs;
		}
	}

	return result;
}

// Scalar Subtraction
template<typename T>
Matrix<T> Matrix<T>::operator-(const T& rhs) {
	Matrix<T> result = Matrix(this->m_rows, this->m_cols, 0.0);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; this->m_cols; j++) {
			result(i,j) = this->mat[i][j] - rhs;
		}
	}

	return result;
}

// Scalar Multiplication
template<typename T>
Matrix<T> Matrix<T>::operator*(const T& rhs) {
	Matrix<T> result = Matrix(this->m_rows, this->m_cols, 0.0);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; this->m_cols; j++) {
			result(i,j) = this->mat[i][j] * rhs;
		}
	}
	return result;
}

// Scalar Division
template<typename T>
Matrix<T> Matrix<T>::operator/(const T& rhs) {
	Matrix<T> result = Matrix(this->m_rows, this->m_cols, 0.0);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; this->m_cols; j++) {
			result(i,j) = this->mat[i][j] / rhs;
		}
	}

	return result;
}

// Multiply matrix with vector
template<typename T>
std::vector<T> Matrix<T>::operator*(const std::vector<T>& rhs) {
	if (this->m_cols != rhs.size()) {
		throw MatrixException(__FILE__, __LINE__, __PRETTY_FUNCTION__, "Error: Matrix columns and vector size must be equal for multiplication");
	}
	std::vector<T> result(this->m_rows, 0.0);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; this->m_cols; j++) {
			result[i] = this->mat[i][j] * rhs[j];
		}
	}

	return result;
}

// Obtain a vector of the diagonal elements
template<typename T>
std::vector<T> Matrix<T>::diagVec() {
	std::vector<T> result(this->m_rows, 0.0);

	for (unsigned i=0; i < this->m_rows; i++) {
		result[i] = this->mat[i][i];
	}

	return result;
}

// Access the individual elements
template<typename T>
T& Matrix<T>::operator()(const unsigned& row, const unsigned& col) {
	return this->mat[row][col];
}

// Access the individual elements (const)
template<typename T>
const T& Matrix<T>::operator()(const unsigned& row, const unsigned& col) const {
	return this->mat[row][col];
}

// Get number of rows of the matrix
template<typename T>
 unsigned Matrix<T>::getRows() const {
	return this->m_rows;
}

// Get number of columns of the matrix
template<typename T>
unsigned Matrix<T>::getCols() const {
	return this->m_cols;
}

// Get size of the matrix
template<typename T>
unsigned long long Matrix<T>::getSize() const {
	return this->m_size;
}

// Explicit Instantiations
template class Matrix<int>;
template class Matrix<short>;
template class Matrix<long>;
template class Matrix<long long>;
template class Matrix<unsigned>;
template class Matrix<unsigned short>;
template class Matrix<unsigned long>;
template class Matrix<unsigned long long>;
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<long double>;
