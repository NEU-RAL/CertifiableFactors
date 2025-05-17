//
// Created by jason on 2/4/25.
//

#ifndef STIEFELMANIFOLDEXAMPLE_LIFTEDPOSEPRIORFACTOR_H
#define STIEFELMANIFOLDEXAMPLE_LIFTEDPOSEPRIORFACTOR_H


/**
 * @file   LiftedPosePriorFactor.h
 * @date
 * @author Jason Xu
 * @brief  Prior for LiftedPose
 */

#pragma once

#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include "StiefelManifold.h"
#include "StiefelManifold-inl.h"
#include "LiftedPose.h"
#include <type_traits>
namespace gtsam {

    /**
     * @brief A prior factor on a lifted pose variable in a factor graph.
     *
     * A `LiftedPosePriorFactor` imposes a Gaussian prior on a lifted pose
     * variable of dynamic dimension.  It inherits from
     * `NoiseModelFactorN<LiftedPoseDP>`, allowing you to specify an N-ary
     * noise model for the prior.
     *
     * @tparam d  The dimensionality of the lifted pose (e.g., 2 for 2D, 3 for 3D).
     */
    template <size_t d>
    class GTSAM_EXPORT LiftedPosePriorFactor : public NoiseModelFactorN<LiftedPoseDP> {
        /// Measured pose matrix between two poses.
        Matrix M_;

        /// Number of rows (p) in the lifted-pose representation.
        size_t p_;

        /// Number of columns (d) in the lifted-pose representation.
        size_t d_;

        /// Cached product of p_ and d_ for fast indexing.
        size_t pd_;

    public:

        // Provide access to the Matrix& version of evaluateError:
        using NoiseModelFactor1<LiftedPoseDP>::evaluateError;

        /// @name Constructor
        /// @{

        /**
         * @brief Constructs a prior factor on a lifted pose variable.
         *
         * Initializes the factor with a key, measured lifted pose, ambient dimension,
         * and noise model. Verifies that the noise model dimension matches expected.
         *
         * @param j1     Key for the variable this factor is attached to.
         * @param Y1     Measured lifted pose (matrix form).
         * @param p      Ambient dimension p of the Stiefel manifold.
         * @param model  Shared noise model for this factor.
         * @throws std::invalid_argument if model dimension ≠ d*p + p.
         */
        LiftedPosePriorFactor(Key j1, const LiftedPoseDP &Y1, size_t p,
                                   const SharedNoiseModel &model = nullptr);

        /// @}
        /// @name Testable
        /// @{

        /**
         * @brief Prints the factor for debugging.
         *
         * Outputs a prefix string, the factor type, its key, measurement matrix M_,
         * and the noise model parameters.
         *
         * @param s             Optional prefix string.
         * @param keyFormatter  Function to format keys for printing.
         */
        void  print(const std::string &s, const KeyFormatter &keyFormatter = DefaultKeyFormatter) const override;

        /**
         * @brief Checks approximate equality to another factor.
         *
         * Compares this factor to `expected` by type, noise model, dimension p_,
         * and measurement matrix M_ within a tolerance.
         *
         * @param expected  The factor to compare against.
         * @param tol       Absolute tolerance for comparisons.
         * @return          True if factors are equal within tol.
         */
        bool equals(const NonlinearFactor &expected, double tol = 1e-9) const override;

        /// @}
        /// @name NoiseModelFactorN methods
        /// @{


        /**
         * @brief Evaluates the error vector for the prior factor.
         *
         * Computes the difference between the stored measurement M_ and the
         * current lifted pose Q1.matrix(), optionally computes Jacobians.
         *
         * @param Q1   Current lifted pose variable.
         * @param H1   Optional pointer to receive Jacobians.
         * @return     Error vector of length d*p + p.
         * @throws std::invalid_argument if Q1.matrix() has incorrect row dimension.
         */
        Vector evaluateError(const LiftedPoseDP& Y1, OptionalMatrixType H1) const override;
        /// @}

//    private:
        /**
         * @brief Fills the Jacobian matrix H1 for this factor.
         *
         * Computes ∂error/∂liftedPose and writes into H1 if provided.
         *
         * @param Q1   Current lifted pose variable.
         * @param H1   Optional pointer to preallocated Jacobian matrix.
         */
        void fillJacobians(const LiftedPoseDP& Y1, OptionalMatrixType H1) const;

    };

// Explicit instantiation for d=2 and d=3 in .cpp file:
    using LiftedPosePriorFactor2 = LiftedPosePriorFactor<2>;
    using LiftedPosePriorFactor3 = LiftedPosePriorFactor<3>;

} // namespace gtsam



#endif //STIEFELMANIFOLDEXAMPLE_LIFTEDPOSEPRIORFACTOR_H
