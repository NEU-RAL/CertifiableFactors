//
// Created by jason on 1/12/25.
//
/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010-2019, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file RaFactor.cpp
 * @date
 * @author Jason Xu, Nikolas Sanderson
 * @brief
 */

#include <gtsam/base/timing.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/FrobeniusFactor.h>
#include "RaFactor.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
namespace gtsam {

//******************************************************************************
    template <size_t d>
    /**
     * @brief Constructs a rotation-averaging factor between two Stiefel manifold variables.
     *
     * This factor enforces the measured relative rotation between two poses, represented
     * as points on a Stiefel manifold of size p × d. It inherits from
     * NoiseModelFactorN<StiefelManifoldKP, StiefelManifoldKP>, allowing specification
     * of an N-ary noise model for the measurement.
     *
     * @tparam d  Dimensionality of the rotation component (rows of the Stiefel manifold).
     * @param j1  Key corresponding to the first pose variable in the graph.
     * @param j2  Key corresponding to the second pose variable in the graph.
     * @param R12 The measured relative rotation from pose j1 to pose j2.
     * @param p   Number of columns in the Stiefel manifold representation.
     * @param model Shared pointer to the noise model encoding measurement uncertainty.
     *
     * @throws std::invalid_argument if the noise model dimension does not equal d * p.
     */
    RaFactor<d>::RaFactor(Key j1, Key j2, const Rot &R12, size_t p,
                                  const SharedNoiseModel &model)
            : NoiseModelFactorN<StiefelManifoldKP, StiefelManifoldKP>(model, j1, j2),
              M_(R12.matrix()), // d*d in all cases
              p_(p),            //
              pd_(p * d) {
        if (noiseModel()->dim() != d * p_)
            throw std::invalid_argument(
                    "RaFactor: model with incorrect dimension.");
        // k used in Stiefel manifold, actually are equivalent to d used in RaFactor
    }

//******************************************************************************
    template <size_t d>
    /**
     * @brief Print a human-readable representation of the RaFactor.
     *
     * Outputs the factor’s identifier, its associated keys, the measured rotation
     * matrix, and the noise model to standard output.
     *
     * @param s             A prefix string printed before the factor information.
     * @param keyFormatter  A callable that formats Keys into strings for display.
     */
    void RaFactor<d>::print(const std::string &s,
                                const KeyFormatter &keyFormatter) const {
        std::cout << s << "RaFactor<" << p_ << ">(" << keyFormatter(key<1>()) << ","
                  << keyFormatter(key<2>()) << ")\n";
        traits<Matrix>::Print(M_, "  M: ");
        noiseModel_->print("  noise model: ");
    }

//******************************************************************************
    template <size_t d>
    /**
     * @brief Determine whether this RaFactor is equal to another NonlinearFactor.
     *
     * This function performs a type-safe comparison by dynamic casting the input
     * factor to RaFactor<d>. It then delegates to the base class's equals method
     * for the noise model factor comparison and additionally checks that the
     * manifold size `p_` and measured rotation matrix `M_` match exactly.
     *
     * @param expected  The other factor to compare against.
     * @param tol       Numerical tolerance for comparison in the base class check.
     * @return `true` if `expected` is a RaFactor<d> with the same noise model,
     *         the same `p_` value, and an identical `M_` matrix; `false` otherwise.
     */
    bool RaFactor<d>::equals(const NonlinearFactor &expected,
                                 double tol) const {
        auto e = dynamic_cast<const RaFactor *>(&expected);
        return e != nullptr && NoiseModelFactorN<StiefelManifoldKP , StiefelManifoldKP>::equals(*e, tol) &&
               p_ == e->p_ && M_ == e->M_;
    }

//******************************************************************************
    template <size_t d>
    /**
     * @brief Compute the Jacobians of the rotation-averaging factor with respect to its two Stiefel manifold inputs.
     *
     * This method computes the partial derivatives of the factor error function
     * with respect to Q1 and Q2. If the optional output matrices H1 or H2 are
     * provided (non-null), they will be resized and populated with the
     * corresponding Jacobian blocks.
     *
     * The Jacobian with respect to Q1 is given by:
     *   H1 = -(I_p ⊗ M) * G1
     * where M is the measured rotation matrix, I_p is the p×p identity, and
     * G1 is the basis matrix stored in Q1.
     *
     * The Jacobian with respect to Q2 is given by:
     *   H2 = I_{pd} * G2
     * where I_{pd} is the pd×pd identity and G2 is the basis matrix in Q2.
     *
     * @tparam d        Dimensionality of the rotation component (rows of the Stiefel manifold).
     * @param Q1        First Stiefel manifold variable.
     * @param Q2        Second Stiefel manifold variable.
     * @param H1        Optional pointer to the output Jacobian matrix w.r.t. Q1.
     *                  If non-null, this matrix is resized to (p*d)×Dim(Q1) and filled.
     * @param H2        Optional pointer to the output Jacobian matrix w.r.t. Q2.
     *                  If non-null, this matrix is resized to (p*d)×Dim(Q2) and filled.
     */
    void RaFactor<d>::fillJacobians(const StiefelManifoldKP &Q1, const StiefelManifoldKP &Q2,
                                        OptionalMatrixType H1,
                                        OptionalMatrixType H2) const {
        const size_t dim = p_ * d; // Stiefel manifold dimension
        const Matrix M1 = Q1.matrix();
        const Matrix M2 = Q2.matrix();


        if (H1) {
            // If asked, calculate Jacobian H as -(I_p \otimes M) * G
            // M = dxd, I_p = pxp, G = (d*p)xDim(p), result should be dim x Dim(p)
            H1->resize(dim, Q1.G_.cols()); // Initialize H with zeros

            Matrix dF_dM1 = Matrix::Zero(p_ * d, p_ * d);
            Matrix I_p = Matrix::Identity(p_, p_);

            for (int i = 0; i < d; ++i) {
                for (int j = 0; j < d; ++j) {
                    dF_dM1.block(i * p_, j * p_, p_, p_) -= M_(j, i) * I_p;
                }
            }
            *H1 = dF_dM1 *Q1.G_;


        }
        if (H2) {
            //
            H2->resize(dim, Q2.G_.cols());
            Matrix dF_dM2 = Matrix::Identity(p_*d, p_*d);
            *H2 = dF_dM2 * Q2.G_;
        }

    }
//******************************************************************************
    template <size_t d>
    /**
     * @brief Compute the error vector for the rotation-averaging factor.
     *
     * This function evaluates the residual between the second Stiefel manifold
     * variable Q2 and the rotated first variable Q1 * M_. The error is given by:
     *   error = vec(Q2) - vec(Q1 * M_)
     * where vec(·) stacks the matrix columns into a single vector of length p*d.
     * If requested, the optional Jacobians H1 and H2 are computed via fillJacobians.
     *
     * @tparam d  Dimensionality of the rotation component (number of columns in Q).
     * @param Q1  First Stiefel manifold variable (p×d matrix).
     * @param Q2  Second Stiefel manifold variable (p×d matrix).
     * @param H1  Optional pointer to the Jacobian w.r.t. Q1; if non-null, it will be
     *            resized to (p*d)×Dim(Q1) and populated.
     * @param H2  Optional pointer to the Jacobian w.r.t. Q2; if non-null, it will be
     *            resized to (p*d)×Dim(Q2) and populated.
     * @return    A Vector of length p*d representing the error.
     *
     * @throws std::invalid_argument if the row dimensions of Q1 or Q2 do not match p_.
     */
    Vector RaFactor<d>::evaluateError(const StiefelManifoldKP &Q1, const StiefelManifoldKP &Q2,
                                          OptionalMatrixType H1,
                                          OptionalMatrixType H2) const {

        const Matrix &M1 = Q1.matrix();
        const Matrix &M2 = Q2.matrix();
        if (M1.rows() != static_cast<int>(p_) || M2.rows() != static_cast<int>(p_))
            throw std::invalid_argument("Invalid dimension SOn values passed to "
                                        "ShonanFactor<d>::evaluateError");

        const size_t dim = p_ * d; // Stiefel manifold dimension
        Vector fQ2(dim), hQ1(dim);

        //
        fQ2 << Eigen::Map<const Matrix>(M2.data(), dim, 1);

        //
        const Matrix Q1R12 = M1 * M_;
        hQ1 << Eigen::Map<const Matrix>(Q1R12.data(), dim, 1);

        this->fillJacobians(Q1, Q2, H1, H2);

        return fQ2 - hQ1;
    }

/* ************************************************************************* */
// Explicit instantiation for d=2 and d=3
    template class RaFactor<2>;
    template class RaFactor<3>;

//******************************************************************************

} // namespace gtsam
