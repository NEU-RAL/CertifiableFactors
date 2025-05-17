//
// Created by jason on 1/11/25.
//
/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010-2019, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   RaFactor.h
 * @date
 * @author Jason Xu, Nikolas Sanderson
 * @brief  Rotation Averaging factor of StiefelManifold type
 */

#pragma once

#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include "StiefelManifold.h"
#include "StiefelManifold-inl.h"
#include <Eigen/Sparse>
#include <gtsam/inference/Symbol.h>
#include <type_traits>
namespace gtsam {

    /**
     * @brief Factor for rotation averaging between two Stiefel manifold variables.
     *
     * `RaFactor` enforces a measured relative rotation between two poses on a
     * Stiefel manifold of size p × d. It inherits from
     * `NoiseModelFactorN<StiefelManifoldKP, StiefelManifoldKP>`, allowing you to
     * specify a noise model for the measurement.
     *
     * @tparam d  Ambient dimension: 2 for Rot2, 3 for Rot3.
     */
    template <size_t d>
class GTSAM_EXPORT RaFactor : public NoiseModelFactorN<StiefelManifoldKP , StiefelManifoldKP> {
        /// Measured rotation matrix between pose 1 and pose 2 (d × d).
        Matrix M_;

        /// Number of columns in the Stiefel manifold representation (p).
        size_t p_;

        /// Dimensionality constant (k), used in Stiefel manifold traits (equals d).
        size_t k_;

        /// Cached product of p_ and d for fast indexing (p_ * d).
        size_t pd_;


        /**
         * @brief Alias for the appropriate rotation type.
         *
         * Selects `Rot2` when `d == 2`, otherwise `Rot3` when `d == 3`.
         */
        using Rot = typename std::conditional<d == 2, Rot2, Rot3>::type;
    public:

        // Provide access to the Matrix& version of evaluateError:
        using NoiseModelFactor2<StiefelManifoldKP, StiefelManifoldKP>::evaluateError;

        /// @name Constructor
        /// @{

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
        RaFactor(Key j1, Key j2, const Rot &R12, size_t p,
                     const SharedNoiseModel &model = nullptr);

        /// @}
        /// @name Testable
        /// @{

        /**
         * @brief Print a human-readable representation of the RaFactor.
         *
         * Outputs the factor’s identifier, its associated keys, the measured rotation
         * matrix, and the noise model to standard output.
         *
         * @param s             A prefix string printed before the factor information.
         * @param keyFormatter  A callable that formats Keys into strings for display.
         */
        void  print(const std::string &s, const KeyFormatter &keyFormatter = DefaultKeyFormatter) const override;

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
        bool equals(const NonlinearFactor &expected, double tol = 1e-9) const override;

        /// @}
        /// @name NoiseModelFactorN methods
        /// @{

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
        Vector evaluateError(const StiefelManifoldKP& S1, const StiefelManifoldKP& S2, OptionalMatrixType H1, OptionalMatrixType H2) const override;
        /// @}

//    private:
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
        void fillJacobians(const StiefelManifoldKP &Q1, const StiefelManifoldKP &Q2,
                           OptionalMatrixType H1,
                           OptionalMatrixType H2) const;

        /**
         * @brief Build the sparse Laplacian block for the rotation connectivity measurement.
         *
         * This method constructs the block of the connectivity Laplacian matrix
         * corresponding to the rotation measurement between two poses identified
         * by key1() and key2(). It produces diagonal entries for each pose and
         * off-diagonal entries weighted by the measured rotation matrix.
         *
         * @tparam Scalar  Numeric scalar type for matrix entries.
         * @return A std::vector of Eigen::Triplet<Scalar> containing the non-zero
         *         entries of the Laplacian block in (row, col, value) form.
         *
         * @note  Assumes the factor’s noise model is a Diagonal noise model.
         */
        std::vector<Eigen::Triplet<Scalar>> getRotationConnLaplacianBlock() {
            std::vector<Eigen::Triplet<Scalar>> triplets;
            size_t measurement_stride = 2 * (d + d * d);
            triplets.reserve(measurement_stride);
            size_t i = gtsam::symbolIndex(this->key1());
            size_t j = gtsam::symbolIndex(this->key2());
            auto diag   = std::dynamic_pointer_cast<noiseModel::Diagonal>(this->noiseModel());
            const Vector& sigmas = diag->sigmas();
            double sigma_kappa = sigmas(0);
            double kappa  = 1.0 / (2*sigma_kappa*sigma_kappa);
            // Elements of ith block-diagonal
            for (size_t k = 0; k < d; k++)
            triplets.emplace_back(d * i + k, d * i + k,
                                  kappa);

            // Elements of jth block-diagonal
            for (size_t k = 0; k < d; k++)
            triplets.emplace_back(d * j + k, d * j + k,
                                  kappa);

            // Elements of ij block
            for (size_t r = 0; r < d; r++)
            for (size_t c = 0; c < d; c++)
            triplets.emplace_back(i * d + r, j * d + c,
            -kappa *
            this->M_(r, c));

            // Elements of ji block
            for (size_t r = 0; r < d; r++)
            for (size_t c = 0; c < d; c++)
            triplets.emplace_back(j * d + r, i * d + c,
            -kappa *
            this->M_(c, r));

            return triplets;
    }

    };

// Explicit instantiation for d=2 and d=3 in .cpp file:
    using RaFactor2 = RaFactor<2>;
    using RaFactor3 = RaFactor<3>;

} // namespace gtsam
