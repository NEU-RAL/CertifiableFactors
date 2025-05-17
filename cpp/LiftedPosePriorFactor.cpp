/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file LiftedPosePriorFactor.cpp
 * @date
 * @author Jason Xu
 * @brief
 */

#include <gtsam/base/timing.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/FrobeniusFactor.h>
#include "StiefelManifoldPriorFactor.h"
#include <cmath>
#include <iostream>
#include <vector>
#include "LiftedPosePriorFactor.h"

using namespace std;
namespace gtsam {

//******************************************************************************
    template <size_t d>
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
    LiftedPosePriorFactor<d>::LiftedPosePriorFactor(
        Key j1,
        const LiftedPoseDP& Y1,
        size_t p,
        const SharedNoiseModel& model)
        : NoiseModelFactorN<LiftedPoseDP>(model, j1),
          M_(Y1.matrix()),
          p_(p),
          d_(d),
          pd_(p * d) {
          if (noiseModel()->dim() != d * p_ + p_) {
            throw std::invalid_argument(
                "LiftedPosePriorFactor: model with incorrect dimension.");
          }
    }

//******************************************************************************
    template <size_t d>
    /**
     * @brief Prints the factor for debugging.
     *
     * Outputs a prefix string, the factor type, its key, measurement matrix M_,
     * and the noise model parameters.
     *
     * @param s             Optional prefix string.
     * @param keyFormatter  Function to format keys for printing.
     */
    void LiftedPosePriorFactor<d>::print(
        const std::string& s,
        const KeyFormatter& keyFormatter) const {
          std::cout << s << "LiftedPosePriorFactor<" << p_ << ">("
                    << keyFormatter(key<1>()) << ")\n";
          traits<Matrix>::Print(M_, "  M: ");
          noiseModel_->print("  noise model: ");
    }

//******************************************************************************
template <size_t d>
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
    bool LiftedPosePriorFactor<d>::equals(
        const NonlinearFactor& expected,
        double tol) const {
          auto e = dynamic_cast<const LiftedPosePriorFactor*>(&expected);
          return e != nullptr
              && NoiseModelFactorN<LiftedPoseDP>::equals(*e, tol)
              && p_ == e->p_
              && M_ == e->M_;
        }

//******************************************************************************
    template <size_t d>
    /**
     * @brief Fills the Jacobian matrix H1 for this factor.
     *
     * Computes ∂error/∂liftedPose and writes into H1 if provided.
     *
     * @param Q1   Current lifted pose variable.
     * @param H1   Optional pointer to preallocated Jacobian matrix.
     */
    void LiftedPosePriorFactor<d>::fillJacobians(
        const LiftedPoseDP& Q1,
        OptionalMatrixType H1) const {
          const size_t dim    = p_ * d + p_;
          const auto   Y1     = Q1.get_StiefelElement();
          const size_t St_dim = StiefelManifoldKP::Dimension(d_, p_);
          const size_t t_dim  = p_;

          if (H1) {
            H1->resize(dim, St_dim + t_dim);
            H1->setZero();
              // Block for the Stiefel manifold part
              H1->block(0, 0, p_ * d, St_dim) =
                      -Matrix::Identity(p_ * d, p_ * d) * Y1.G_;
              // Block for the translation part
              H1->block(p_ * d, St_dim, p_, t_dim) =
                      -Matrix::Identity(p_, t_dim);
          }
    }

//******************************************************************************
template <size_t d>
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
    Vector LiftedPosePriorFactor<d>::evaluateError(
        const LiftedPoseDP& Q1,
        OptionalMatrixType H1) const {
      const auto& M1 = Q1.matrix();
          if (M1.rows() != static_cast<int>(p_)) {
            throw std::invalid_argument(
                "Invalid dimension passed to LiftedPosePriorFactor::evaluateError");
          }
          this->fillJacobians(Q1, H1);
          // Flatten M_ and M1 and compute their difference
          return Eigen::Map<const Matrix>(M_.data(), p_*d + p_, 1)
               - Eigen::Map<const Matrix>(M1.data(), p_*d + p_, 1);
        }


//******************************************************************************
// Explicit instantiations for d = 2 and d = 3
template class LiftedPosePriorFactor<2>;
template class LiftedPosePriorFactor<3>;


} // namespace gtsam
