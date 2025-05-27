/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson.
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#pragma once
#include "types.h"
#include <gtsam/base/Manifold.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/dllexport.h>
#include <gtsam/base/Vector.h>
#include <iostream>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <random>
#include <cmath>
#include <gtsam/nonlinear/Values.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <Eigen/CholmodSupport>
#include <Eigen/Geometry>
#include <Eigen/SPQRSupport>

namespace gtsam{

/**
 * @brief Helper function of calculating dimension
 */
namespace internal {
// Calculate dimensionality of Stiefel manifold St(k, p) at compile time, or return Dynamic if invalid inputs
// St(k, p) is in R^(p x k), where p >= k
    constexpr int DimensionSt(int k, int p) {
        return (k < 0 || p < 0 || p < k) ? Eigen::Dynamic : k * (p - k) + (k * (k - 1)) / 2;
    }
// Calculate k * p (ambient Euclidean space dimension) at compile time, or return Dynamic if invalid inputs
    constexpr int KTimesPSt(int k, int p) {
        return (k < 0 || p < 0 || p < k) ? Eigen::Dynamic : k * p;
    }
}  // namespace internal

/**
 * @brief This struct contains definition and calculation related to Tangent space of Stiefel Manifold
 */
struct TangentSpace {
    // Matrix of the Stiefel manifold
    Eigen::MatrixXd Y;
    // Basis for the tangent space
    std::vector<Eigen::MatrixXd> basis;
    // Dimension
    size_t dim;

    // Default Constructor
    TangentSpace()
            : Y(Eigen::MatrixXd::Zero(0, 0)),
              basis(),
              dim(0) {}

    /**
    * @brief Constructor
     *
    * In GTSAM, most of the variables types are based on Lie group, so the tangent space(Lie algebra) is just vector.
    * But for the more general manifolds, the tangent space is of matrix. So to make the general tangent matrix
    * compatible with the tangent vector of GTSAM, we choose to represent tangent matrix as tangent vector with basis.
    * The basis are just the matrices that satisfy the constraints of tangent space. See detail of basis definition
    * in computeTangentBasis().
     *
    * @param Y_in  Matrix of the Stiefel manifold
    */
    TangentSpace(const Eigen::MatrixXd& Y_in) {
        if (!isOnStiefelManifold(Y_in)) {
            throw std::invalid_argument("Y is not on the Stiefel manifold (Y.transpose() * Y != I_k)");
        }
        Y = Y_in;
        dim = Y.cols() * (Y.rows() - Y.cols()) + (Y.cols() * (Y.cols() - 1)) / 2;
        computeTangentBasis();
    }

    /**
      * @brief Check whether it is a Stiefel manifold.
      * @param Y    Matrix of the Stiefel manifold
      * @param tol  Numerical tolerance of float point computation
      * @return Ture if the given matrix is belong to Stiefel manifold(satisfy the constraints)
      */
    static bool isOnStiefelManifold(const Eigen::MatrixXd& Y, double tol = 1e-6) {
        if (Y.rows() < Y.cols()) {
            std::cerr << "Matrix is too wide to be column-orthonormal (rows < cols)" << std::endl;
            return false;
        }
        Eigen::MatrixXd gram = Y.transpose() * Y;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(gram.rows(), gram.cols());
        double diff = (gram - I).norm();

        return diff < tol;
    }

    /**
      * @brief Check whether it satisfies the constraints of tangent space
      * @param V    Matrix of the tangent space element
      * @param tol  Numerical tolerance of float point computation
      * @return Ture id the given matrix is in the tangent space of corresponding Stiefel manifold
      */
    bool isInTangentSpace(const Eigen::MatrixXd& V, double tol = 1e-6) const {
        if (V.rows() != Y.rows() || V.cols() != Y.cols()) {
            throw std::invalid_argument("V dimensions do not match Y dimensions.");
        }
        // Compute the constraint
        Eigen::MatrixXd constraint = Y.transpose() * V + V.transpose() * Y;
        auto norm = constraint.norm();

        // Check if the constraint is approximately zero
        return norm < tol;
    }

    /**
      * @brief Generates unit skew symmetric basis A ( A^-1 =  A^T), of the  dimension d
      * @return The vector of unit skew-symmetric basis
      */
    std::vector<Eigen::MatrixXd> generateSkewBasis()
    {
        int k = Y.cols();
        std::vector<Eigen::MatrixXd> skewBasis;
        int num = -1;
        for (int i = 0; i < k; ++i) {
            for (int j = i + 1; j < k; ++j) {
                // Create a P x P zero matrix
                Eigen::MatrixXd A = Eigen::MatrixXd::Zero(k,k);

                // Set the skew-symmetric values
                A(i,j) = num;
                A(j,i) = -num;

                // Add the matrix to the basis
                skewBasis.push_back(A);

                // Alternate the sign
                num *= -1;
            }
        }
        return skewBasis;
    }

    /**
      * @brief Generate the orthogonal complement of Stiefle manifold matrix Y
      * @return  The orthogonal complement of Y (i.e., Y^T * Y_perp = 0, Y_perp^T * Y_perp = I_(n-p)
      */
    Eigen::MatrixXd generateXPerp() {
        // Perform QR decomposition of Y
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(Y);

        // Full orthonormal basis
        Eigen::MatrixXd Q = qr.householderQ();

        // Extract the last (n - p) columns of Q for X_perp
        int n = Y.rows();
        int p = Y.cols();
        Eigen::MatrixXd Y_perp = Q.block(0, p, n, n - p);
        return Y_perp;
    }

    /**
      * @brief Compute the tangent space basis
      */
    void computeTangentBasis() {
        int p = Y.rows();
        int k = Y.cols();

        // Step 1: Compute the orthogonal complement Q of Y in R^n
        Eigen::MatrixXd Q = generateXPerp();

        // Step 2: Create basis for skew-symmetric matrices (A)
        std::vector<Eigen::MatrixXd> skewBasis = generateSkewBasis();
        
        // Step 3: Combine Y @ A and Q @ B to form the tangent basis
        basis.clear();
        // 3.1, Add Y @ A terms to the basis
        for (const auto& A : skewBasis) {
            Eigen::MatrixXd tangentVector = (Y * A).normalized();
            basis.push_back(tangentVector);
        }

        // 3.2, Add Q @ B terms to the basis
            for (int j = 0; j < (p-k)*k; ++j) {
                Eigen::VectorXd Bv = Eigen::VectorXd::Zero((p-k)*k);
                Bv(j) = 1;
                Eigen::MatrixXd B = DeVectorize(Bv,p-k,k);
                // Compute the tangent vector
                Eigen::MatrixXd tangentVector = (Q * B).normalized();
                basis.push_back(tangentVector);
            }
    }

    /**
      * @brief Generate a random tangent matrix (mainly used in unit test)
      * @return Matrix of the corresponding tangent space
      */
    Eigen::MatrixXd randomTangentVector() const {
        if (basis.empty()) {
            throw std::runtime_error("Tangent basis is empty. Ensure the tangent basis is created.");
        }

        // Generate random coefficients
        Eigen::VectorXd coefficients = Eigen::VectorXd::Random(basis.size());

        // Represent the matrix as a linear combination of basis matrices:
        //     randomVector = Σ_i coefficients[i] * basis[i]
        // where each basis[i] is a matrix and coefficients[i] is its scalar weight.
        Eigen::MatrixXd randomVector = Eigen::MatrixXd::Zero(Y.rows(), Y.cols());
        for (size_t i = 0; i < basis.size(); ++i) {
            randomVector += coefficients[i] * basis[i];
        }

        return randomVector;
    }

    /**
      * @brief Reshape a vector into a (rows x cols) matrix.
      *
      * Performs the inverse of vectorization by mapping the input vector
      * to a matrix in column-major order.
      *
      * @param vector Flattened matrix.
      * @param rows Target number of rows.
      * @param cols Target number of columns.
      * @return Reshaped matrix.
      */
    Matrix DeVectorize(const Vector& vector, const int rows, const int cols) const {
            if (vector.size() != rows * cols) {
                throw std::invalid_argument("Vector size does not match matrix dimensions.");
            }
            return Eigen::Map<const Matrix>(vector.data(), rows, cols);
    }
    /**
      * @brief Verify tangent space constraints (mainly used in unit test)
      */
    void verifyTangentConstraints() const {
        std::cout << "\nVerifying tangent constraints (Y.transpose() * Xi + Xi.transpose() * Y = 0):" << std::endl;
        for (size_t i = 0; i < basis.size(); ++i) {
            Eigen::MatrixXd constraint = Y.transpose() * basis[i] + basis[i].transpose() * Y;
            double norm = constraint.norm();
            std::cout << "Basis vector " << i + 1 << ": Norm of constraint violation = " << norm << std::endl;
        }
    }
};

    template <int K, int P>
    class GTSAM_EXPORT StiefelManifold;

    using StiefelManifoldKP = StiefelManifold<Eigen::Dynamic, Eigen::Dynamic>;

// GTSAM traits
    template<int K, int P>
    struct traits<StiefelManifold<K, P>> : public internal::Manifold<StiefelManifold<K, P>> {};

    template<int K, int P>
    struct traits<const StiefelManifold<K, P>> : public internal::Manifold<StiefelManifold<K, P>> {};

/**
 * @brief Represents the Stiefel manifold St(K, P).
 *
 * The Stiefel manifold St(K, P) is the set of all orthonormal K-frames
 * (i.e., K orthonormal vectors) in P-dimensional Euclidean space.
 * This class provides a lightweight representation of an element on the
 * Stiefel manifold and related operations needed for a variable definition of GTSAM, like: retract
 *
 * @tparam K Number of orthonormal vectors in the frame (K is sometimes also denoted as d, e.g., the problem dimension, like SO2, SO3...)
 * @tparam P Dimension of the ambient Euclidean space.
 */
template <int K, int P>
class GTSAM_EXPORT StiefelManifold {
    private:
        // Number of vectors in orthonormal K-frame
        size_t k_; // Default initialized to 0
        // Dimension of ambient Euclidean space containing the frames
        size_t p_; // Default initialized to 0

    public:
        // Stiefel manifold dimension
        inline constexpr static auto dimension = internal::DimensionSt(K, P);
        using MatrixPK = Eigen::Matrix<double, P, K>;

        // Ambient Euclidean dimension
        using VectorKTimesP = Eigen::Matrix<double, internal::KTimesPSt(K, P), 1>;

        // Basis generator for tangent space
        TangentSpace T_x;
        Matrix G_;

    protected:
        // The matrix in ambient Euclidean dimension
        MatrixPK matrix_;
//        template<int K_, int P_>
//        using IsValid = typename std::enable_if<(K_ > 0 || P_ > 0), void>::type;
        template<int K_, int P_>
        using IsValid = typename std::enable_if<
                (K_ > 0 || P_ > 0) && (K_ != Eigen::Dynamic && P_ != Eigen::Dynamic), void>::type;

        template<int K_, int P_>
        using IsDynamic = typename std::enable_if<
                (K_ == Eigen::Dynamic || P_ == Eigen::Dynamic), void>::type;
    public:
    /// @name Constructors
    /// @{
        // Default constructor
        template<int K_ = K, int P_ = P, typename  = IsValid<K_, P_>>
        StiefelManifold() : k_(K_), p_(P_), matrix_(MatrixPK::Identity()) {
            initializeTangentSpace();
        }

        // NOn-templated default constructor
        // A true zero‐arg ctor for *all* K,P that simply picks a safe default (mainly for UnitSphere)
        StiefelManifold()
                : k_( (K==Eigen::Dynamic) ? 1 : K ),
                  p_( (P==Eigen::Dynamic) ? 1 : P ),
                  matrix_( MatrixPK::Identity( p_, k_ ) )
        {
            initializeTangentSpace();
        }

        /// Construct identity for N == Eigen::Dynamic
        template <int K_ = K, int P_ = P, typename = IsDynamic<K_, P_>>
        StiefelManifold(size_t k = 0, size_t p = 0): k_(k), p_(p), matrix_(Eigen::MatrixXd::Identity(p, k)) {
            initializeTangentSpace();
        }

        // Constructor from Eigen Matrix, dynamic version
        template <typename Derived>
        explicit StiefelManifold(const Eigen::MatrixBase<Derived>& R) : k_(R.cols()), p_(R.rows()) ,matrix_(R.eval()) {
            initializeTangentSpace();
        }

        // Named constructor from Eigen Matrix
        template <typename Derived>
        static StiefelManifold FromMatrix(const Eigen::MatrixBase<Derived>& R) {
            return StiefelManifold(R);
        }

        // Construct dynamic StiefelManifold(K, P) from Fixed SO<M, N>
        template <int M, int N, int K_ = K, int P_ = P, typename = IsDynamic<K_, P_>>
        explicit StiefelManifold(const StiefelManifold<M, N>& R) : k_(R.cols()), p_(R.rows()), matrix_(R.matrix()) {
            initializeTangentSpace();
        }

        // Named constructor from lower dimensional matrix
        template <typename Derived, int K_ = K, int P_ = P, typename = IsDynamic<K_, P_>>
        static StiefelManifold Lift(size_t p_, const Eigen::MatrixBase<Derived> &R) {
            Matrix Q = Matrix::Zero(p_, R.cols());
            Q.topLeftCorner(R.rows(), R.cols()) = R;
            return StiefelManifold(Q);
        }

    /// @}
    /// @name Standard methods
    /// @{
        // Setters
        void set_k(size_t k) { k_ = k; }
        void set_p(size_t p) { p_ = p; }
        void incrementRank() { p_++; }
        void setRank(size_t p) { p_ = p; }

        // Getters
        size_t get_k() const { return k_; }
        size_t get_p() const { return p_; }

        // Returm matrix
        const MatrixPK& matrix() const {return matrix_; };

        size_t rows() const { return matrix_.rows(); }
        size_t cols() const { return matrix_.cols(); }

    /// @}
    /// @name Testable
    /// @{
        void print(const std::string& s = std::string()) const;

        bool equals(const StiefelManifold& other, double tol) const {
            return equal_with_abs_tol(matrix_, other.matrix_, tol);
        }

    /// @}
    /// @name Manifold
    /// @{

        using TangentVector = Eigen::Matrix<double, dimension, 1>;

        static size_t Dimension(size_t k, size_t p) {return  k * (p - k) + (k * (k - 1)) / 2;}

        // Calculate run-time dimensionality of manifold.
        // Available as dimension or Dim() for fixed N.
        size_t dim() const { return k_ * (p_ - k_) + (k_ * (k_ - 1)) / 2; }

        /**
          * @brief Projection of A in R^{p x k} onto R.
          *
          * Closest in the Frobenius norm sense. From SE-Sync:
          * "We use a generalization of the well-known SVD-based projection for the
          * orthogonal and special orthogonal groups; see for example Proposition 7
          * in the paper "Projection-Like Retractions on Matrix Manifolds" by Absil
          * and Malick"
          *
          * @param A Matrix of the ambient Euclidean space
          * @return Stiefel manifold element
          */
        static StiefelManifold<K, P> projectToManifold(const Matrix &A);

        /**
         * @brief Retraction given a tangent vector
         *
         * Given an element Y in M and a tangent vector V in T_Y(M), this function
         * computes the retraction along V at Y using the QR-based retraction
         * specified in eq. (4.8) of Absil et al.'s  "Optimization Algorithms on
         * Matrix Manifolds").
         *
         * @param V tangent vector in T_Y(M)
         * @return Updated variable in manifold
         */
        StiefelManifold<K, P> retractMatrix(const Matrix &V) const
            {
                return projectToManifold(matrix_ + V);
            }

       /**
         * @brief Compute the product with symmetric block diagonal element
         *
         * Returns product of P = A * SymBlockDiag(B^T * C)
         * Definition of SymBlockDiag operator refer to equation(5) of SE-Sync report
         * A, B, C are {p x k} matrices
         *
         * @param A matrix A
         * @param BT the transpose of matrix B
         * @param C matrix C
         * @return A * SymBlockDiag(B^T * C)
         */
        static Matrix SymBlockDiagProduct(const gtsam::Matrix &A, const gtsam::Matrix &BT, const gtsam::Matrix &C);

        /**
          * @brief Project the element in tangent space ambient Euclidean space to
          * the tangent space of Stiefel manifold
          *
          * Given an element Y in M and a matrix V in T_X(R^{p x kn}) (that is, a (p
          * x kn)-dimensional matrix V considered as an element of the tangent space to
          * the *entire* ambient Euclidean space at X), this function computes and
          * returns the projection of V onto T_X(M), the tangent space of M at X (cf.
          * eq. (42) in the SE-Sync tech report).
          *
          * @param V element in the tangent space of the ambient Euclidean space at X
          * @return element in the tangent space of M
          */
        Matrix projectToTangentSpace(const Matrix &V) const
        {
            return V - SymBlockDiagProduct(matrix_,matrix_.transpose(),V);
        }

        /**
          * @brief Static veriosn of projectToTangentSpace
          * the tangent space of Stiefel manifold
          * @param Y matrix of the stiefel manifold
          * @param V element in the tangent space of the ambient Euclidean space at X
          * @return element in the tangent space of M
          */
        static Matrix Proj(const Matrix &Y, const Matrix &V) {
            return V - SymBlockDiagProduct(Y, Y.transpose(), V);
        }

        /**
         * @brief Placeholder to satisfy GTSAM variable interface.
         *
         * Not needed for current usage, but required to conform to GTSAM's
         * variable concept.
         */
        TangentVector localCoordinates(const StiefelManifold& R) const
        {
            throw std::runtime_error("StiefelManifold::Logmap is not implemented.");
        }

        /**
          * @brief Retraction methods for the gtsam variable type
          *
          * Just retraction. But we use tangent vector to fulfill the "Lie group-like" tangent space definition of GTSAM,
          * actual tangent matrix can be uniquely represented as a vector of its coordinates of the basis.
          *
          * @param V update in the tangent space(using tangent vector form here, given specific basis)
          * @return updated element of Stiefel manifold
          */
        StiefelManifold retract(const TangentVector& V) const
        {
            // 1, Given the tangent vector, multiply with the vectorized basis generator M, obtain the vectorized tangent matrix
            // Then, recover the tangent matrix through de-vectorize.
            const Matrix M = DeVectorize(G_ * V, p_, k_);
            // 2, Retract with the tangent matrix
            return retractMatrix(M);
        }

    /// @}
    /// @name Other methods
    /// @{

        /**
          * @brief Generate the (vectorized) basis
          *
          * Each columns is the vectorized basis. G is in R^(dim(St) x dim(ambient Euclidean space)).
          * i.e., R^(dim(St) x (k*p)), dim(St) is the dimension of stiefel manifold element
          *
          * @return Basis
          */
        Matrix BasisGenerators() const {
            const size_t PK = p_ * k_, dim = Dimension(k_, p_);
            Matrix G(PK, dim);
            for (size_t i = 0; i < dim; i++) {
                G.col(i) = Eigen::Map<const VectorKTimesP>(T_x.basis[i].data(), PK, 1);
            }
            return G;
        }

        // returns a vector from a matrix
        static Vector Vectorize(Matrix &M)
        {
            const size_t dim = M.rows() * M.cols();
            return Eigen::Map<const Matrix>(M.data(), dim, 1);
        }

        // returns a matrix from a vector
        static Matrix DeVectorize(const Vector& vector, const int rows, const int cols) {
            if (vector.size() != rows * cols) {
                throw std::invalid_argument("Vector size does not match matrix dimensions.");
            }
            return Eigen::Map<const Matrix>(vector.data(), rows, cols);
        }

        /**
          * @brief Generate a random sample on the Stiefel manifold St(K, P).
          *
          * Generates a random orthonormal K-frame in P-dimensional space using the given seed.
          *
          * @param seed Random seed for reproducibility (optional).
          * @return A random StiefelManifold<K, P> instance.
          */
        StiefelManifold<K, P> random_sample(const std::default_random_engine::result_type &seed = std::default_random_engine::default_seed) const;

        /**
         * @brief Generate a random Stiefel manifold element St(K, P).
         *
         * Generates a random orthonormal K-frame in P-dimensional space.
         * We define the k and p, mainly for static case.
         * Allows optional specification of k_ and p_ (used for dynamic variants).
         *
         * @param seed Random seed for reproducibility (optional).
         * @param k_ Optional number of vectors (for dynamic use, defaults to 0).
         * @param p_ Optional ambient dimension (for dynamic use, defaults to 0).
         * @return A random StiefelManifold<K, P> instance.
         */
        static StiefelManifold<K, P> Random(const std::default_random_engine::result_type &seed = std::default_random_engine::default_seed, size_t k_ = 0, size_t p_ = 0);

        /**
         * @brief Initialize the tangent space and basis at the current point.
         *
         * Constructs the tangent space basis T_x at the current point represented by matrix_,
         * and initializes G_ with the corresponding basis generators.
         */
        void initializeTangentSpace() {
            T_x = TangentSpace(matrix_);
            G_ = BasisGenerators();
        }

        /**
         * @brief Lift the current Stiefel point to a higher-dimensional ambient space.
         *
         * Pads the current (P x K) matrix to (p_lifted x K) by zero-padding the lower rows,
         * effectively embedding the point in a higher-dimensional Euclidean space.
         *
         * @param p_lifted Target ambient dimension (must be >= current rows).
         * @return Lifted StiefelManifold in the new space.
         * @throws std::runtime_error if p_lifted is smaller than current rows.
         */
        StiefelManifold LiftTo(size_t p_lifted) const {
            if (p_lifted < this->rows())
                throw std::runtime_error("LiftTo: target p must be greater than or equal to current rows.");
            Matrix Q = Matrix::Zero(p_lifted, this->cols());
            Q.topLeftCorner(this->rows(), this->cols()) = this->matrix();
            return StiefelManifold(Q);
        }

    /// @}

};

} // namespace gtsam

