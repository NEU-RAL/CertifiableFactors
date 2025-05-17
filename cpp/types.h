//
// Created by jason on 1/27/25.
//
/** A set of typedefs describing the types of matrices and factorizations that
 * will be used in the SE-Sync algorithm.
 *
 * Copyright (C) 2016 - 2018 by David M. Rosen (dmrosen@mit.edu)
 */

#ifndef STIEFELMANIFOLDEXAMPLE_TYPES_H
#define STIEFELMANIFOLDEXAMPLE_TYPES_H
#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> DiagonalMatrix;

/** We use row-major storage order to take advantage of fast (sparse-matrix) *
 * (dense-vector) multiplications when OpenMP is available (cf. the Eigen
 * documentation page on "Eigen and Multithreading") */
typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;



#endif //STIEFELMANIFOLDEXAMPLE_TYPES_H