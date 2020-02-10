/*
 * MatrixException.h
 *
 *  Created on: Feb 8, 2020
 *      Author: jwredhead
 */

#ifndef MATRIXEXCEPTION_H_
#define MATRIXEXCEPTION_H_

#include <iostream>
#include <exception>

class MatrixException : public std::exception{



public:
	MatrixException(const char* file_, int line_, const char* func_, const char* info_ = "") :
		file (file_),
		line (line_),
		func (func_),
		info (info_)
	{
	}

	const char* getFile() const {return file; }
	int getLine() const {return line; }
	const char* getFunc() const {return func; }
	const char* getInfo() const {return info; }

private:
	const char* file;
	int line;
	const char* func;
	const char* info;

};

#endif /* MATRIXEXCEPTION_H_ */
