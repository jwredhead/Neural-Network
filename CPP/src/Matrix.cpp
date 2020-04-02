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
Matrix<T>::Matrix(unsigned rows, unsigned cols) : m_rows(rows), m_cols(cols) {
	m_size = rows * cols;
	mat = new T[m_size];
	for (unsigned i=0; i < m_size; i++) {
		mat[i] = 0.0;
	}
}

// NxN Constructor
template<typename T>
Matrix<T>::Matrix(unsigned rows) : m_rows(rows), m_cols(rows){
	m_size = rows * rows;
	mat = new T[m_size];
	for (unsigned i=0; i < m_size; i++) {
		mat[i] = 0.0;
	}
}

// Empty Constructor
template<typename T>
Matrix<T>::Matrix() : m_rows(0), m_cols(0), m_size(0) {
	mat = nullptr;
}

// Array Constructor - Converts array into matrix
template<typename T>
Matrix<T>::Matrix(T* arr, unsigned size) {
	m_cols = 1;
	m_rows = size;
	m_size = size;

	mat = new T(m_size);
	for (unsigned i=0; i < m_size; i++) {
		mat[i] = arr[i];
	}

}

// Virtual Destructor, set to default
template<typename T>
Matrix<T>::~Matrix() {
	if(mat != nullptr)
	{
		delete mat;
	}
	mat = nullptr;
}

// Assign Constructor
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T> &rhs) {
	if (&rhs == this) return *this;

	if (this->mat != nullptr) {
		delete this->mat;
		this->mat = nullptr;
	}

	this->m_rows = rhs.getRows();
	this->m_cols = rhs.getCols();
	this->m_size = rhs.getSize();

	this->mat = new T[m_size];
    for(unsigned i=0; i < this->getRows(); i++) {
    	for(unsigned j=0; j < this->getCols(); j++) {
    		this->mat[i*(this->m_cols)+j] = rhs.mat[i*(rhs.m_cols)+j];
    	}
    }

	return *this;
}

// Copy Constructor
template<typename T>
Matrix<T>::Matrix(const Matrix<T> &rhs) {
	if(&rhs == this)

	if(this->mat != nullptr) {
		delete this->mat;
		this->mat = nullptr;
	}

	this->m_rows = rhs.getRows();
	this->m_cols = rhs.getCols();
	this->m_size = rhs.getSize();

	this->mat = new T[m_size];
    for(unsigned i=0; i < this->getRows(); i++) {
    	for(unsigned j=0; j < this->getCols(); j++) {
    		this->mat[i*(this->m_cols)+j] = rhs.mat[i*(rhs.m_cols)+j];
    	}
    }
}

// Access the individual elements
template<typename T>
T& Matrix<T>::operator()(const unsigned& row, const unsigned& col) {
	return this->mat[row*m_cols+col];
}

// Access the individual elements (const)
template<typename T>
const T& Matrix<T>::operator()(const unsigned& row, const unsigned& col) const {
	return this->mat[row*(this->m_cols)+col];
}

// Addition Operator
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &rhs) {
	if (this->m_rows != rhs.getRows() || this->m_cols != rhs.getCols()) {
		throw MatrixException(__FILE__, __LINE__, __PRETTY_FUNCTION__, "Error: Matrix rows and columns must be equal for matrix addition");
	}

	Matrix<T> result = Matrix(this->m_rows, this->m_cols);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; j < this->m_cols; j++) {
			result(i,j) = this->mat[i*(this->m_cols)+j] + rhs(i,j);
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
		for (unsigned j=0; j < this->m_cols; j++) {
			this->mat[i*(this->m_cols)+j] = this->mat[i*(this->m_cols)+j] + rhs(i,j);
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

	Matrix<T> result = Matrix(this->m_rows, this->m_cols);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; j < this->m_cols; j++) {
			result(i,j) = this->mat[i*(this->m_cols)+j] - rhs(i,j);
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
		for (unsigned j=0; j < this->m_cols; j++) {
			this->mat[i*(this->m_cols)+j] = this->mat[i*(this->m_cols)+j] - rhs(i,j);
		}
	}

	return *this;
}

// Multiplication Operator
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &rhs) {
	unsigned rhsRows = rhs.getRows();
	unsigned rhsCols = rhs.getCols();

	if (this->m_cols != rhsRows) {
		throw MatrixException(__FILE__, __LINE__, __PRETTY_FUNCTION__, "Error: Rows of Matrix A and columns of Matrix B must be equal for matrix multiplication");
	}

	Matrix<T> result = Matrix(this->m_rows, rhsCols);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; j < rhsCols; j++) {
			for (unsigned k=0 ; k < this->m_cols; k++) {
				result(i,j) += this->mat[i*(this->m_cols)+k] * rhs(k,j);
			}
		}
	}

	return result;
}

// Cumulative Multiplication Operator
template<typename T>
Matrix<T> Matrix<T>::operator*=(const Matrix<T> &rhs) {
	unsigned rhsRows = rhs.getRows();

	if (this->m_cols != rhsRows) {
		throw MatrixException(__FILE__, __LINE__, __PRETTY_FUNCTION__, "Error: Matrix rows and columns must be equal for matrix multiplication");
	}

	Matrix<T> result = Matrix(this->m_rows, rhs.getCols());

	(*this) = (*this) * rhs;

	return *this;
}

// Hardamard Product
template<typename T>
Matrix<T> Matrix<T>::hadamardProduct(const Matrix<T> &rhs) {
	if (this->m_rows != rhs.getRows() || this->m_cols != rhs.getCols()) {
		throw MatrixException(__FILE__, __LINE__, __PRETTY_FUNCTION__, "Error: Matrix rows and columns must be equal for Hardamard Product");
	}

	Matrix<T> result = Matrix(this->m_rows, this->m_cols);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; j< this->m_cols; j++) {
			result(i,j) = this->mat[i*(this->m_cols)+j] * rhs(i,j);
		}
	}

	return result;
}

// Transpose
template<typename T>
Matrix<T> Matrix<T>::transpose() {
	Matrix<T> result = Matrix(this->m_cols, this->m_rows);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; j< this->m_cols; j++) {
			result(j,i) =this->mat[i*(this->m_cols)+j];
		}
	}

	return result;
}

// Fill
template<typename T>
Matrix<T> Matrix<T>::fill(T val) {
	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; j< this->m_cols; j++) {
			this->mat[i*(this->m_cols) + j] = val;
		}
	}
	return *this;
}

// Scalar Addition
template<typename T>
Matrix<T> Matrix<T>::operator+(const T& rhs) {
	Matrix<T> result = Matrix(this->m_rows, this->m_cols);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; j < this->m_cols; j++) {
			result(i,j) = this->mat[i*(this->m_cols)+j] + rhs;
		}
	}

	return result;
}

// Scalar Subtraction
template<typename T>
Matrix<T> Matrix<T>::operator-(const T& rhs) {
	Matrix<T> result = Matrix(this->m_rows, this->m_cols);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; j < this->m_cols; j++) {
			result(i,j) = this->mat[i*(this->m_cols)+j] - rhs;
		}
	}

	return result;
}

// Scalar Multiplication
template<typename T>
Matrix<T> Matrix<T>::operator*(const T& rhs) {
	Matrix<T> result = Matrix(this->m_rows, this->m_cols);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; j < this->m_cols; j++) {
			result(i,j) = this->mat[i*(this->m_cols)+j] * rhs;
		}
	}
	return result;
}

// Scalar Division
template<typename T>
Matrix<T> Matrix<T>::operator/(const T& rhs) {
	Matrix<T> result = Matrix(this->m_rows, this->m_cols);

	for (unsigned i=0; i < this->m_rows; i++) {
		for (unsigned j=0; j < this->m_cols; j++) {
			result(i,j) = this->mat[i*(this->m_cols)+j] / rhs;
		}
	}

	return result;
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
