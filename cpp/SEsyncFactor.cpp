/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */
#include <gtsam/base/timing.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include "SEsyncFactor.h"
#include <iostream>
#include <vector>

using namespace std;
namespace gtsam {

//******************************************************************************
    template <size_t d>
    /**
     * @brief Construct a synchronization factor on SE(d) between two lifted poses.
     *
     * This factor encodes the measured relative rotation and translation
     * between two lifted pose variables in the factor graph. It inherits
     * from NoiseModelFactorN<LiftedPoseDP, LiftedPoseDP> to incorporate
     * a Gaussian noise model for the SE(d) measurement.
     *
     * @tparam d       The spatial dimension (2 for 2D, 3 for 3D).
     * @param j1       Key of the first lifted pose variable.
     * @param j2       Key of the second lifted pose variable.
     * @param R12      Measured rotation matrix between the two poses.
     * @param T12      Measured translation vector between the two poses.
     * @param p        Number of columns in the lifted-pose representation.
     * @param model    Shared pointer to the noise model encoding measurement uncertainty.
     *
     * @throws std::invalid_argument if the noise model dimension does not equal d * p + p.
     */
    SEsyncFactor<d>::SEsyncFactor(Key j1, Key j2, const Matrix &R12, const Vector &T12 , size_t p,
                          const SharedNoiseModel &model)
            : NoiseModelFactorN<LiftedPoseDP, LiftedPoseDP>(model, j1, j2),
              M_(R12), //
              V_(T12),
              p_(p),
              d_(d), //
              pd_(p * d) {
        if (noiseModel()->dim() != ((d * p_) + p_))
            throw std::invalid_argument(
                    "SEsyncFactor: model with incorrect dimension.");
    }

//******************************************************************************
    template <size_t d>
    /**
     * @brief Print a human-readable representation of the SEsyncFactor.
     *
     * Outputs the factor’s identifier including its template parameters,
     * the associated variable keys, the measured rotation matrix, the
     * measured translation vector, and the noise model to standard output.
     *
     * @tparam d  Spatial dimension of the synchronization (2 or 3).
     * @param s             A prefix string printed before the factor information.
     * @param keyFormatter  A callable that formats Keys into strings for display.
     */
    void SEsyncFactor<d>::print(const std::string &s,
                            const KeyFormatter &keyFormatter) const {
        std::cout << s << "SEsyncFactor<" << p_ << " * " << d << ">(" << keyFormatter(key<1>()) << ","
                  << keyFormatter(key<2>()) << ")\n";
        traits<Matrix>::Print(M_, "  M: ");
        traits<Matrix>::Print(V_, "  V: ");
        noiseModel_->print("  noise model: ");
    }

//******************************************************************************
    template <size_t d>
    /**
     * @brief Check equality between this SEsyncFactor and another NonlinearFactor.
     *
     * Performs a type-safe comparison by dynamic casting the input factor to
     * SEsyncFactor<d>. Delegates to the base class’s equals method to compare
     * the noise model and then checks that the dimensionality `p_`, rotation
     * matrix `M_`, and translation vector `V_` match exactly.
     *
     * @tparam d     Spatial dimension of the synchronization (2 or 3).
     * @param expected  The other factor to compare against.
     * @param tol       Numerical tolerance for the base class comparison.
     * @return `true` if `expected` is an SEsyncFactor<d> with the same noise model,
     *         the same `p_`, identical `M_` and `V_`; `false` otherwise.
     */
    bool SEsyncFactor<d>::equals(const NonlinearFactor &expected,
                             double tol) const {
        auto e = dynamic_cast<const SEsyncFactor *>(&expected);
        return e != nullptr && NoiseModelFactorN<LiftedPoseDP , LiftedPoseDP>::equals(*e, tol) &&
               p_ == e->p_ && M_ == e->M_ && V_ == e->V_;
    }

//******************************************************************************
    template <size_t d>
    /**
     * @brief Compute the Jacobians of the SEsyncFactor error with respect to each lifted pose variable.
     *
     * This method computes and, if requested, populates the Jacobian matrices H1 and H2
     * for the SE synchronization factor. The error consists of both the Stiefel manifold
     * component (rotation) and the translation component. H1 and H2 are resized and filled
     * to represent the partial derivatives of the error with respect to Q1 and Q2, respectively.
     *
     * @tparam d      Spatial dimension of the synchronization (2 for 2D, 3 for 3D).
     * @param Q1      The first lifted pose variable, providing Y1 (Stiefel manifold) and t1 (translation).
     * @param Q2      The second lifted pose variable, providing Y2 and t2.
     * @param H1      Optional pointer to the output Jacobian w.r.t. Q1. If non-null, this matrix
     *                is resized to ((p_*d_ + p_) × (Dim(Y1) + p_)) and populated.
     * @param H2      Optional pointer to the output Jacobian w.r.t. Q2. If non-null, this matrix
     *                is resized to ((p_*d_ + p_) × (Dim(Y2) + p_)) and populated.
     */
    void SEsyncFactor<d>::fillJacobians(const LiftedPoseDP &Q1, const LiftedPoseDP &Q2,
                                    OptionalMatrixType H1,
                                    OptionalMatrixType H2) const {
        const StiefelManifoldKP Y1 = Q1.get_Y();
        const StiefelManifoldKP Y2 = Q2.get_Y();
        const Vector t1 = Q1.get_t();
        const Vector t2 = Q2.get_t();
        const size_t St_OutDim = p_ * d_;
        const size_t St_dim = StiefelManifoldKP::Dimension(d_, p_); // Better use Dimension of some dim function later
        const size_t t_OutDim = p_;
        const size_t t_dim = p_;
        const Matrix M1 = Y1.matrix();
        const Matrix M2 = Y2.matrix();


        if (H1) {
            // If asked, calculate Jacobian H as -(I_p \otimes M) * G
            // M = dxd, I_p = pxp, G = (d*p)xDim(p), result should be dim x Dim(p)
//            H1->resize(St_dim, Y1.G_.cols()); // Initialize H with zeros
            H1->resize(St_OutDim + t_OutDim, Y1.G_.cols() + p_);
            H1->setZero();
            Matrix dF_dM1 = Matrix::Zero(p_ * d, p_ * d);
            Matrix I_p = Matrix::Identity(p_, p_);

            for (int i = 0; i < d; ++i) {
                for (int j = 0; j < d; ++j) {
                    dF_dM1.block(i * p_, j * p_, p_, p_) -= M_(j, i) * I_p;
                }
            }
            H1->block(0, 0, St_OutDim, St_dim) = dF_dM1 * Y1.G_;
                // Initialize J as a zero matrix of size (p, p * d)
                Matrix dt_dY = Matrix::Zero(t_OutDim, St_OutDim);
                // Fill J using the explicit Kronecker product expansion
//                for (int i = 0; i < p_; ++i) {
//                    for (int j = 0; j < d_; ++j) {
//                        dt_dY(i, i * d + j) = -V_(j);  // Each row gets a copy of -T^T
//                    }
//                    dt_dY.block(i, i * d_, 1, d_) = -V_.transpose();
//                    dt_dY.row(i) = -V_.transpose();
//                }
            for (int j = 0; j < d; ++j) {
                dt_dY.block(0, j * p_, p_, p_) -= V_(j) * Matrix ::Identity(p_, p_);
            }
            H1->block(St_OutDim, 0, t_OutDim, St_dim) = dt_dY * Y1.G_;
            H1->block(St_OutDim, St_dim, t_OutDim, t_dim) =
                    -Matrix::Identity(t_OutDim, t_dim);


        }
        if (H2) {
            //
            H2->resize(St_OutDim + t_OutDim, Y2.G_.cols() + p_);
            H2->setZero();
            Matrix dF_dM2 = Matrix::Identity(p_*d, p_*d);
            H2->block(0, 0, St_OutDim, St_dim) = dF_dM2 * Y2.G_;
            H2->block(St_OutDim, St_dim, t_OutDim, t_dim) =
                    Matrix::Identity(t_OutDim, t_dim);

        }

    }
//******************************************************************************
    template <size_t d>
    /**
     * @brief Compute the error vector for the SE synchronization factor.
     *
     * The error consists of two parts:
     *   1. Rotation residual: vec(Y2) − vec(Y1 * M_)
     *   2. Translation residual: t2 − t1 − Y1 * V_
     * The full error is the concatenation of these two residuals.
     *
     * If requested, the Jacobians H1 and H2 are computed via fillJacobians.
     *
     * @tparam d      Spatial dimension of the synchronization (2 or 3).
     * @param Q1      First lifted pose variable, providing Y1 (Stiefel manifold) and t1 (translation).
     * @param Q2      Second lifted pose variable, providing Y2 and t2.
     * @param H1      Optional pointer to the Jacobian w.r.t. Q1; if non-null, it will be resized
     *                and populated with the partial derivatives of the error w.r.t. Q1.
     * @param H2      Optional pointer to the Jacobian w.r.t. Q2; if non-null, it will be resized
     *                and populated with the partial derivatives of the error w.r.t. Q2.
     * @return        A Vector of length (p_*d + p_) containing the stacked rotation and translation errors.
     *
     * @throws std::invalid_argument if the row dimensions of Y1 or Y2 do not match p_.
     */
    Vector SEsyncFactor<d>::evaluateError(const LiftedPoseDP &Q1, const LiftedPoseDP &Q2,
                                      OptionalMatrixType H1,
                                      OptionalMatrixType H2) const {

        const StiefelManifoldKP Y1 = Q1.get_Y();
        const StiefelManifoldKP Y2 = Q2.get_Y();
        const Vector t1 = Q1.get_t();
        const Vector t2 = Q2.get_t();


        const Matrix &M1 = Y1.matrix();
        const Matrix &M2 = Y2.matrix();
        if (M1.rows() != static_cast<int>(p_) || M2.rows() != static_cast<int>(p_))
            throw std::invalid_argument("Invalid dimension SOn values passed to "
                                        "SEsyncFactor<d>::evaluateError");

        // Error of rotation part
        const size_t St_dim = p_ * d; // Stiefel manifold dimension
        Vector fQ2(St_dim), hQ1(St_dim), error_rotation(St_dim);
        fQ2 << Eigen::Map<const Matrix>(M2.data(), St_dim, 1);
        const Matrix Q1R12 = M1 * M_;
        hQ1 << Eigen::Map<const Matrix>(Q1R12.data(), St_dim, 1);
        error_rotation = fQ2 - hQ1;

        // Error of translation part
        const size_t trans_dim = p_; // lifted translation dimension ( R^d -> R^p)
        Vector error_trans(trans_dim);
//        error_trans = LiftedPoseDP::LiftToRp(t2, p_) -  LiftedPoseDP::LiftToRp(t1, p_) - M1 * V_;
        error_trans = t2 -  t1 - M1 * V_;

        this->fillJacobians(Q1, Q2, H1, H2);

        Vector error(St_dim + trans_dim);
        error << error_rotation, error_trans;

        return error;
    }

/* ************************************************************************* */
// Explicit instantiation for d=2 and d=3
    template class SEsyncFactor<2>;
    template class SEsyncFactor<3>;

//******************************************************************************

} // namespace gtsam
