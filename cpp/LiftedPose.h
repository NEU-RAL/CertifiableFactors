/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */
#ifndef STIEFELMANIFOLDEXAMPLE_LIFTEDPOSE_H
#define STIEFELMANIFOLDEXAMPLE_LIFTEDPOSE_H

#include <gtsam/base/Matrix.h>
#include <gtsam/dllexport.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Point2.h>
#include <iostream>
#include <random>
#include <cmath>
#include <type_traits>
#include "StiefelManifold.h"
#include "StiefelManifold-inl.h"
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <variant>
namespace gtsam
{
    namespace internal
    {

    /**
     * @brief Compute the total dimension of the lifted pose variable [Y | t].
     *
     * Y is Stiefel manifold St(d, p) embedding in R^(pxd) , t is lifted position t (R^d -> R^t)
     *
     * Returns Eigen::Dynamic if inputs are invalid (p < d or negative).
     *
     * @param d  Ambient manifold dimension.
     * @param p  Relaxation rank (lifted dimension).
     * @return   Dimensionality of [Y|t], or Eigen::Dynamic on error.
     */
        constexpr int DimensionLiftPose(int d, int p) {
            return (d < 0 || p < 0 || p < d) ? Eigen::Dynamic : d * (p - d) + (d * (d - 1)) / 2 + p;
        }

    }  // namespace internal


    //=============================================================================
    /// @brief Forward declaration of the LiftedPose class template.
    ///
    /// @tparam D  Compile-time ambient dimension.
    /// @tparam P  Compile-time lifted dimension (rank).
    template<int D, int P>
    class GTSAM_EXPORT LiftedPose;

    //=============================================================================
    /// @brief Alias for a row-major sparse matrix using the Scalar type.
    typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;

    //=============================================================================
    /// @brief Alias for a dynamically sized LiftedPose (both ambient and rank dynamic).
    using LiftedPoseDP = LiftedPose<Eigen::Dynamic, Eigen::Dynamic>;

    //=============================================================================
    /// @brief Traits specialization to integrate LiftedPose<D,P> with GTSAM’s manifold framework.
    ///
    /// This enables the use of manifold operations (retraction, local coordinates, etc.)
    /// on LiftedPose instances within GTSAM algorithms.
    ///
    /// @tparam D  Ambient dimension.
    /// @tparam P  Lifted dimension (rank).
    template <int D, int P>
    struct traits<LiftedPose<D, P>>
      : public internal::Manifold<LiftedPose<D, P>> {};

    //=============================================================================
    /// @brief Const-qualified traits specialization for LiftedPose<D,P>.
    ///
    /// Ensures that const LiftedPose types also satisfy the manifold interface.
    ///
    /// @tparam D  Ambient dimension.
    /// @tparam P  Lifted dimension (rank).
    template <int D, int P>
    struct traits<const LiftedPose<D, P>>
      : public internal::Manifold<LiftedPose<D, P>> {};


    // P, D is the same as the Stiefel element
    template<int D, int P>
    class GTSAM_EXPORT LiftedPose{
    private:
        // Number of rows in the combined matrix [Y | t].
        size_t Rows_;

        // Number of columns in the combined matrix [Y | t].
        size_t Cols_;

        // Ambient dimension (P) and manifold dimension (D) stored at runtime if dynamic.
        size_t p_;
        size_t d_;

        /**
         * @brief SFINAE helper: only defined if either template parameter is dynamic.
         *
         * Allows compile‐time enablement of methods that only make sense when
         * D or P equals Eigen::Dynamic.
         */
        template<int D_, int P_>
        using IsDynamic =
            typename std::enable_if<(D_ == Eigen::Dynamic || P_ == Eigen::Dynamic), void>::type;

        // Dynamic‐sized matrix type to hold [Y | t].
        using MatrixDP = Eigen::MatrixXd;

        // Combined matrix storing the Stiefel element Y and translation t.
        MatrixDP matrix_;

        // Rotation component on the Stiefel manifold St(P, D).
        StiefelManifoldKP Y;

        // Translation vector in ℝᴰ.
        Vector t;

        // Flattened vector form of [Y | t], useful for certain optimizers.
        Vector liftedVector_;
    public:
        // Compile‑time dimension of the lifted pose: dim(St(d, p)) + p (translation component).
        inline constexpr static auto dimension = internal::DimensionLiftPose(D, P);

        // Alias for the tangent vector type, stored as a dynamic‑size Eigen vector.
        using TangentVector = Vector;

        // Returns the compile‑time dimension of this lifted pose type.
        static int Dim() { return dimension; }

        // Compute the lifted pose dimension at runtime for given d and p.
        // dim = d*(p‑d) + (d*(d‑1))/2 + p
        static size_t Dimension(size_t d, size_t p) {
            return d * (p - d) + (d * (d - 1)) / 2 + p;
        }

        // Returns the dimension for this instance.
        // If both D and P are dynamic, uses stored d_ and p_; otherwise, a compile‑time constant.
        size_t dim() const {
            if constexpr (D == Eigen::Dynamic && P == Eigen::Dynamic) {
                return d_ * (p_ - d_) + (d_ * (d_ - 1)) / 2 + p_;
            } else {
                return D * (P - D) + (D * (D - 1)) / 2 + P;
            }
        }
        /// @name Constructors
        /// @{

        /**
         * @brief Default constructor for dynamic‑sized LiftedPose.
         *
         * This constructor is only enabled when either D or P is Eigen::Dynamic.
         * It initializes the internal matrix to the identity of size p × (d+1),
         * representing an identity rotation on the Stiefel manifold and zero translation.
         *
         * @param d  The manifold dimension (number of rows).
         * @param p  The ambient dimension (number of columns).
         */
        template <int D_ = D, int P_ = P, typename = IsDynamic<D_, P_>>
        explicit LiftedPose(size_t d = 0, size_t p = 0) {
            matrix_ = Eigen::MatrixXd::Identity(p, d + 1);
            d_ = d;
            p_ = p;
        }

        /**
         * @brief Constructs a LiftedPose from a given Eigen matrix.
         *
         * This constructor initializes the internal storage from an Eigen matrix
         * of size p × (d+1), interpreting the left p×d block as the Stiefel element
         * and the last column as the translation vector.
         *
         * @tparam Derived  Eigen matrix type (e.g., Eigen::MatrixXd, block, etc.).
         * @param R         Input matrix [Y | t] to initialize this LiftedPose.
         */
        template <typename Derived>
        explicit LiftedPose(const Eigen::MatrixBase<Derived>& R)
            : Rows_(R.rows()),
              Cols_(R.cols()),
              matrix_(R.eval()) {}

        /**
         * @brief Factory method to create a LiftedPose from an Eigen matrix.
         *
         * This static method delegates to the explicit matrix constructor.
         *
         * @tparam Derived  Eigen matrix type.
         * @param R         Input matrix [Y | t] to construct the LiftedPose.
         * @return          A new LiftedPose initialized from R.
         */
        template <typename Derived>
        static LiftedPose FromMatrix(const Eigen::MatrixBase<Derived>& R) {
            return LiftedPose(R);
        }

        /**
         * Constructs a dynamic-size LiftedPose from a Stiefel manifold element and translation vector.
         * Enabled only when either D or P is Eigen::Dynamic.
         *
         * The resulting internal matrix_ is [Y | t], where:
         *   - Y ∈ St(P, D) is the rotation on the Stiefel manifold
         *   - t ∈ ℝᵈ is the translation vector, lifted into ℝᵖ by zero-padding
         *
         * @tparam D_  Compile-time manifold dimension (may be Eigen::Dynamic).
         * @tparam P_  Compile-time ambient dimension (may be Eigen::Dynamic).
         * @param Y_   Stiefel manifold element (p × d matrix).
         * @param t_   Translation vector in R^P (Lifted translation)
         */
        template <int D_ = D, int P_ = P, typename = IsDynamic<D_, P_>>
        explicit LiftedPose(const StiefelManifoldKP& Y_, const Vector& t_)
                : Rows_(Y_.get_p()),
                  Cols_(Y_.get_k() + 1),
                  p_(Y_.get_p()),
                  d_(Y_.get_k()),
                  liftedVector_(Vector::Zero(Y_.get_p())),
                  matrix_(Matrix::Zero(Y_.get_p(), Y_.get_k() + 1)),
                  Y(Y_),
                  t(t_)
        {
            // Directly assign t_ into the allocated lifted vector
            liftedVector_ = t_;

            // Assemble [Y | t]
            matrix_.block(0, 0, Y_.rows(), Y_.cols()) = Y_.matrix();
            matrix_.col(Cols_ - 1) = liftedVector_;
        }


        /// @}

        /**
         * @brief Projects a higher‑dimensional vector into ℝᵈ by keeping its first d components.
         *
         * Given a p‑dimensional vector Rp, returns the vector of its first d entries.
         *
         * @param Rp  Input vector in ℝᵖ.
         * @param d   Number of dimensions to project onto (must satisfy 0 ≤ d ≤ Rp.size()).
         * @return    Vector of size d containing the first d elements of Rp.
         */
        static Vector ProjectToRd(const Vector& Rp, int d) {
            // Ensure d is valid (uncomment assertion if desired):
            // assert(d <= Rp.size() && "d must be less than or equal to the size of Rp");
            return Rp.head(d);
        }

        /**
         * @brief Lifts a lower‑dimensional vector into ℝᵖ by zero‑padding.
         *
         * Given an input vector in ℝᵈ, returns a p‑dimensional vector whose first d entries
         * match the input and whose remaining entries are zero.
         *
         * @param Rd  Input vector in ℝᵈ.
         * @param p   Target dimension (must satisfy p ≥ Rd.size()).
         * @return    Vector of size p with Rd in its first entries and zeros thereafter.
         */
        static Vector LiftToRp(const Vector& Rd, int p) {
            Vector Rp = Vector::Zero(p);
            Rp.head(Rd.size()) = Rd;
            return Rp;
        }


        /// @name Getters
        /// @{

        /**
         * @brief Returns the full lifted pose matrix [Y | t].
         *
         * This matrix concatenates the Stiefel manifold element Y and the
         * lifted translation vector into a p×(d+1) matrix.
         *
         * @return matrix_  The combined rotation and translation matrix.
         */
        Matrix get_Matrix() const { return matrix_; }

        /**
         * @brief Returns the Stiefel manifold element Y.
         *
         * Y is the rotation component on St(P, D).
         *
         * @return Y  The Stiefel manifold rotation element.
         */
        StiefelManifoldKP get_StiefelElement() const { return Y; }

        /**
         * @brief Returns the translation vector in ℝᵈ.
         *
         * This is the original d-dimensional translation before lifting.
         *
         * @return t  The translation vector.
         */
        Vector get_TranslationVector() const { return t; }

        /**
         * @brief Returns the lifted translation vector in ℝᵖ.
         *
         * This pads the d-dimensional translation with zeros to match p dimensions.
         *
         * @return LiftToRp(t, Rows_)  The padded translation vector.
         */
        Vector get_LiftedTranslationVector() const { return LiftToRp(t, Rows_); }

        /**
         * @brief Returns the number of rows of the lifted pose matrix.
         *
         * This equals the ambient dimension p of the Stiefel element.
         *
         * @return Rows_  Number of rows (p).
         */
        size_t get_Rows() const { return Rows_; }

        /**
         * @brief Returns the number of columns of the lifted pose matrix.
         *
         * This equals d+1, where d is the manifold dimension.
         *
         * @return Cols_  Number of columns (d+1).
         */
        size_t get_Cols() const { return Cols_; }

        /**
         * @brief Const accessor for the internal matrix storage.
         *
         * Provides read-only access to the full [Y | t] matrix.
         *
         * @return matrix_  Reference to the internal matrix.
         */
        const Matrix& matrix() const { return matrix_; }

        /**
         * @brief Const accessor for the Stiefel element Y.
         *
         * @return Y  The rotation component (Stiefel manifold).
         */
        const StiefelManifoldKP get_Y() const { return Y; }

        /**
         * @brief Const accessor for the translation vector t.
         *
         * @return t  The translation in ℝᵈ.
         */
        const Vector get_t() const { return t; }

        /**
         * @brief Const accessor for the lifted translation vector.
         *
         * @return liftedVector_  The translation padded to ℝᵖ.
         */
        const Vector get_liftedVector() const { return liftedVector_; }

        /// @}

        /// @name Testable
        /// @{

        /**
         * @brief Prints the lifted pose matrix with an optional label.
         *
         * Outputs the provided string followed by the [Y | t] matrix to stdout.
         *
         * @param s  Optional prefix string to print before the matrix.
         */
        void print(const std::string& s = std::string()) const {
            std::cout << s << matrix_ << std::endl;
        }

        /**
         * @brief Compares this LiftedPose to another for approximate equality.
         *
         * Uses an element‑wise absolute tolerance check on the internal matrices.
         *
         * @param other  The LiftedPose to compare against.
         * @param tol    Absolute tolerance for matrix element differences.
         * @return       True if all corresponding elements differ by no more than tol.
         */
        bool equals(const LiftedPose& other, double tol) const {
            return equal_with_abs_tol(matrix_, other.matrix_, tol);
        }

        /// @}

        /**
         * @brief Retracts a tangent vector back onto the lifted pose manifold.
         *
         * Applies the Stiefel manifold retraction to the rotation component and
         * updates the translation component by adding the translation part of V.
         *
         * @param V  Tangent vector of size Dim(), where:
         *           - V.head(Y.dim()) is the tangent for the Stiefel element Y
         *           - V.tail(p_) is the translation update in ℝᵖ
         * @return   A new LiftedPose with the retracted rotation and updated translation.
         */
        LiftedPose retract(const TangentVector &V) const {
            // Retract the Stiefel rotation using its manifold retraction method
            StiefelManifoldKP St = Y.retract(V.head(Y.dim()));

            // Update translation by adding the tail of V and construct new LiftedPose
            return LiftedPose(St, t + V.tail(p_));
        }


        /**
         * @brief Converts a flat vector into a matrix by mapping its underlying data.
         *
         * This function creates an Eigen::Map over the input vector's data,
         * interpreting it as a matrix of size rows × cols without copying.
         *
         * @param vector  The input vector containing exactly rows*cols elements.
         * @param rows    The number of rows in the resulting matrix.
         * @param cols    The number of columns in the resulting matrix.
         * @return        An Eigen::Matrix of dimensions rows × cols mapping the vector's data.
         * @throws std::invalid_argument if vector.size() does not equal rows * cols.
         */
        static Matrix DeVectorize(const Vector& vector, int rows, int cols) {
            if (vector.size() != rows * cols) {
                throw std::invalid_argument("Vector size does not match matrix dimensions.");
            }
            return Eigen::Map<const Matrix>(vector.data(), rows, cols);
        }

        /**
          * @brief Computes the tangent (log‑map) coordinates from this pose to another.
          *
          * Unimplemented.
          *
          * @param R  The target LiftedPose.
          * @throws std::runtime_error Always, since Logmap is not implemented.
          */
        TangentVector localCoordinates(const LiftedPose &R) const {
            throw std::runtime_error("LiftedPose::Logmap not implemented.");
        }

        /**
         * @brief Flattens a matrix into a column‑major vector.
         *
         * This method maps the entries of matrix M into a single vector of length
         * (rows × cols) without copying the underlying data.
         *
         * @param M  The input matrix of size rows × cols.
         * @return   A vector of length rows*cols containing M's entries in column‑major order.
         */
        Vector Vectorize(Matrix &M) const {
            const size_t dim = M.rows() * M.cols();
            return Eigen::Map<const Matrix>(M.data(), dim, 1);
        }


        /**
         * Generates a random lifted pose sample using the object's dimensions.
         * @param seed The seed for the random number generator (default: std::default_random_engine::default_seed).
         * @return A LiftedPose with a random rotation on St(P, D) and a random translation lifted to ℝᵖ.
         */
        LiftedPose random_sample(
            const std::default_random_engine::result_type& seed =
                std::default_random_engine::default_seed) const {
            auto Y_random = StiefelManifoldKP::Random(seed, d_, p_);
            auto t_random = Vector::Random(d_);
            return LiftedPose(Y_random, LiftToRp(t_random, p_));
        }

        /**
         * Generates a random lifted pose for specified dimensions.
         * @param seed The seed for the random number generator (default: std::default_random_engine::default_seed).
         * @param d_   The manifold (rotation) dimension.
         * @param p_   The ambient (lifted) dimension.
         * @return A LiftedPose with a random rotation on St(p_, d_) and a random translation lifted to ℝᵖ.
         */
        static LiftedPose Random(
            const std::default_random_engine::result_type& seed =
                std::default_random_engine::default_seed,
            size_t d_ = 0,
            size_t p_ = 0) {
            auto Y_random = StiefelManifoldKP::Random(seed, d_, p_);
            auto t_random = Vector::Random(d_);
            return LiftedPose(Y_random, LiftToRp(t_random, p_));
        }


        /**
         * @brief Lifts the current pose to a higher ambient dimension.
         *
         * This method increases the ambient dimension of both the Stiefel
         * manifold element and the translation by zero‑padding.
         *
         * @param p_lifted The target ambient dimension (p) to lift to. Must be
         *                 greater than or equal to the current number of rows.
         * @returns A new LiftedPose with its rotation and translation lifted to
         *          the specified dimension.
         * @throws std::runtime_error if p_lifted is less than the current Rows_.
         */
        LiftedPose LiftTo(size_t p_lifted) const {
            if (p_lifted < Rows_) {
                throw std::runtime_error(
                    "LiftTo: target p must be greater than or equal to current rows.");
            }
            // Lift the rotation on the Stiefel manifold to the new dimension.
            StiefelManifoldKP Lifted_Y = Y.LiftTo(p_lifted);
            // Zero‑pad the translation vector to match the new dimension.
            Vector Lifted_t = LiftToRp(t, p_lifted);
            return LiftedPose(Lifted_Y, Lifted_t);
        }


    };


}
#endif //STIEFELMANIFOLDEXAMPLE_LIFTEDPOSE_H
