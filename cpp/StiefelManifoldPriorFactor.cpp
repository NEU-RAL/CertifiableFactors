//
// Created by jason on 2/4/25.
//
/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010-2019, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file StiefelManifoldPriorFactor.cpp
 * @date
 * @author JasonXu
 * @brief
 */

#include <gtsam/base/timing.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/FrobeniusFactor.h>
#include "StiefelManifoldPriorFactor.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
namespace gtsam {

//******************************************************************************
    template <size_t d>
    /**
     * @brief Construct a prior factor on a Stiefel manifold variable.
     *
     * This factor imposes a Gaussian prior on a Stiefel manifold variable Y
     * of size pxd. It inherits from `NoiseModelFactorN<StiefelManifoldKP>`
     * to incorporate a noise model for the prior.
     *
     * @tparam d    Ambient dimension of the Stiefel manifold (number of rows).
     * @param j1    Key corresponding to the Stiefel manifold variable in the graph.
     * @param Y1_initial  Initial value of the Stiefel manifold variable (p×d matrix).
     * @param p     Number of columns in the Stiefel manifold representation.
     * @param model Shared pointer to the noise model encoding the prior covariance.
     *
     * @throws std::invalid_argument if the noise model dimension does not equal d * p.
     */
    StiefelManifoldPriorFactor<d>::StiefelManifoldPriorFactor(Key j1, const StiefelManifoldKP &Y1_initial, size_t p,
                          const SharedNoiseModel &model)
            : NoiseModelFactorN<StiefelManifoldKP>(model, j1),
              M_(Y1_initial.matrix()), // d*d in all cases
              p_(p),
              d_(d),
              pd_(p * d) {
        if (noiseModel()->dim() != d * p_)
            throw std::invalid_argument(
                    "StiefelManifoldPriorFactor: model with incorrect dimension.");
    }
//******************************************************************************
    template <size_t d>
    /**
     * @brief Print a human-readable representation of the Stiefel manifold prior factor.
     *
     * Outputs the factor’s identifier including its template parameter (p),
     * the associated variable key, the prior manifold matrix, and the noise model
     * to standard output.
     *
     * @tparam d              Ambient dimension of the Stiefel manifold (number of rows).
     * @param s              A prefix string printed before the factor information.
     * @param keyFormatter   A callable that formats Keys into strings for display.
     */
    void StiefelManifoldPriorFactor<d>::print(const std::string &s,
                            const KeyFormatter &keyFormatter) const {
        std::cout << s << "StiefelManifoldPriorFactor<" << p_ << ">(" << keyFormatter(key<1>()) <<
                    ")\n";
        traits<Matrix>::Print(M_, "  M: ");
        noiseModel_->print("  noise model: ");
    }
//******************************************************************************
    template <size_t d>
    /**
     * @brief Check equality between this StiefelManifoldPriorFactor and another NonlinearFactor.
     *
     * Performs a type-safe comparison by dynamic casting the input factor to
     * `StiefelManifoldPriorFactor<d>`. Delegates to the base class’s `equals`
     * method to compare the noise model, and additionally checks that the
     * column count `p_` and the prior matrix `M_` match exactly.
     *
     * @tparam d        Ambient dimension of the Stiefel manifold (number of rows).
     * @param expected  The other factor to compare against.
     * @param tol       Numerical tolerance for the base class comparison.
     * @return `true` if `expected` is a `StiefelManifoldPriorFactor<d>` with the
     *         same noise model, the same `p_`, and an identical `M_`; `false`
     *         otherwise.
     */
    bool StiefelManifoldPriorFactor<d>::equals(const NonlinearFactor &expected,
                             double tol) const {
        auto e = dynamic_cast<const StiefelManifoldPriorFactor *>(&expected);
        return e != nullptr && NoiseModelFactorN<StiefelManifoldKP>::equals(*e, tol) &&
               p_ == e->p_ && M_ == e->M_;
    }

//******************************************************************************
    template <size_t d>
    /**
     * @brief Compute the Jacobian of the Stiefel manifold prior error w.r.t. the variable.
     *
     * For a prior factor on a Stiefel manifold variable Y, the error is defined as
     *   Y = vec(Y) - vec(M_).
     * The Jacobian of this error with respect to the vectorized manifold variable
     * is simply -I, mapped through the tangent basis G.
     *
     * @tparam d         Ambient dimension of the Stiefel manifold (number of rows of \(Y\)).
     * @param Q1         The Stiefel manifold variable, providing the tangent basis \(G_1\).
     * @param H1         Optional pointer to receive the Jacobian matrix. If non-null,
     *                   it will be resized to \((p_ \times d)\) rows by
     *                   \(\mathrm{Dim}(Q1)\) columns and populated with \(-I * G_1\).
     */
    void StiefelManifoldPriorFactor<d>::fillJacobians(const StiefelManifoldKP &Q1, OptionalMatrixType H1) const {
        const size_t dim = p_ * d; // Stiefel manifold dimension
        const Matrix M1 = Q1.matrix();

        if (H1) {
            H1->resize(dim, Q1.G_.cols());

            Matrix dF_dM1 = Matrix::Zero(p_ * d, p_ * d);
            Matrix I_p = Matrix::Identity(p_, p_);

            for (int i = 0; i < d; ++i) {
                for (int j = 0; j < d; ++j) {
                    dF_dM1.block(i * p_, j * p_, p_, p_) -=  I_p;
                }
            }
//            *H1 = dF_dM1 *Q1.G_;
              *H1 = -Matrix::Identity(dim, dim) * Q1.G_;
        }
    }
//******************************************************************************
    template <size_t d>
    /**
     * @brief Compute the error vector for the Stiefel manifold prior factor.
     *
     * The error is defined as the difference between the prior matrix M_
     * and the current Stiefel manifold variable Y, vectorized:
     *   e(Y) = vec(M_) − vec(Y).
     * If requested, the Jacobian is computed via `fillJacobians`.
     *
     * @tparam d      Ambient dimension of the Stiefel manifold (number of rows).
     * @param Q1      The Stiefel manifold variable, providing the matrix Y and tangent basis.
     * @param H1      Optional pointer to receive the Jacobian matrix. If non-null, it will be
     *                resized to (p_*d) × Dim(Q1) and populated.
     * @return        A Vector of length p_*d representing the stacked difference
     *                between the prior M_ and the variable Y.
     *
     * @throws std::invalid_argument if the row dimension of Q1 does not equal p_.
     */
    Vector StiefelManifoldPriorFactor<d>::evaluateError(const StiefelManifoldKP &Q1,
                                      OptionalMatrixType H1) const {

        const Matrix &M1 = Q1.matrix();
        if (M1.rows() != static_cast<int>(p_))
            throw std::invalid_argument("Invalid dimension StiefelManifold values passed to "
                                        "StiefelManifoldPriorFactor<d>::evaluateError");

        const size_t dim = p_ * d; // Stiefel manifold dimension

        this->fillJacobians(Q1, H1);

        return Eigen::Map<const Matrix>(M_.data(), dim, 1) - Eigen::Map<const Matrix>(M1.data(), dim, 1);
    }

/* ************************************************************************* */
// Explicit instantiation for d=2 and d=3
    template class StiefelManifoldPriorFactor<1>;
    template class StiefelManifoldPriorFactor<2>;
    template class StiefelManifoldPriorFactor<3>;

//******************************************************************************

} // namespace gtsam
