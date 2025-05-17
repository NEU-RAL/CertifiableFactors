/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef STIEFELMANIFOLDEXAMPLE_STIEFELMANIFOLDPRIORFACTOR_H
#define STIEFELMANIFOLDEXAMPLE_STIEFELMANIFOLDPRIORFACTOR_H

/**
 * @file   StiefelManifoldPriorFactor.h
 * @date
 * @author Jason Xu
 * @brief  Prior for Stiefel Manifold Factor
 */

#pragma once

#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include "StiefelManifold.h"
#include "StiefelManifold-inl.h"

#include <type_traits>
namespace gtsam {

/**
 *
 */
    template <size_t d>
    class GTSAM_EXPORT StiefelManifoldPriorFactor : public NoiseModelFactorN<StiefelManifoldKP> {
        Matrix M_;                    ///< measured rotation between R1 and R2
        size_t p_, d_;               ///< dimensionality constants
        size_t pd_;

    public:

        // Provide access to the Matrix& version of evaluateError:
        using NoiseModelFactor1<StiefelManifoldKP>::evaluateError;

        /// @name Constructor
        /// @{

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
        StiefelManifoldPriorFactor(Key j1, const StiefelManifoldKP &Y1, size_t p,
                 const SharedNoiseModel &model = nullptr);

        /// @}
        /// @name Testable
        /// @{

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
        void
        print(const std::string &s,
              const KeyFormatter &keyFormatter = DefaultKeyFormatter) const override;

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
        bool equals(const NonlinearFactor &expected,
                    double tol = 1e-9) const override;

        /// @}
        /// @name NoiseModelFactorN methods
        /// @{

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
        Vector evaluateError(const StiefelManifoldKP& Y1, OptionalMatrixType H1) const override;
        /// @}

//    private:
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
        void fillJacobians(const StiefelManifoldKP& Y1, OptionalMatrixType H1) const;

    };

// Explicit instantiation for d=2 and d=3 in .cpp file:
    using StPriorFactor1 = StiefelManifoldPriorFactor<1>;
    using StPriorFactor2 = StiefelManifoldPriorFactor<2>;
    using StPriorFactor3 = StiefelManifoldPriorFactor<3>;



} // namespace gtsam


#endif //STIEFELMANIFOLDEXAMPLE_STIEFELMANIFOLDPRIORFACTOR_H
