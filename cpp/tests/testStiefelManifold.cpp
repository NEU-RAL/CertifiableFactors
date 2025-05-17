/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include "../StiefelManifold.h"
#include "../StiefelManifold-inl.h"
#include <iostream>
#include <gtsam/base/Matrix.h>
#include <CppUnitLite/TestHarness.h>

using namespace gtsam;

// Test default Constructor
TEST(StiefelManifoldTest, DefaultConstructor) {
StiefelManifold<2, 3> manifold;
    EXPECT_LONGS_EQUAL(manifold.get_k(), 2);
    EXPECT_LONGS_EQUAL(manifold.get_p(), 3);
    EXPECT_LONGS_EQUAL(manifold.matrix().rows(), 3);
    EXPECT_LONGS_EQUAL(manifold.matrix().cols(), 2);
}

// Test matrix constructor
TEST(StiefelManifoldTest, MatrixConstructor) {
    Eigen::MatrixXd mat;
    mat.resize(3, 2);
    mat << 1, 0,
            0, 1,
            0, 0;
    StiefelManifold<2, 3> manifold(mat);
    EXPECT_LONGS_EQUAL(manifold.get_k(), 2);
    EXPECT_LONGS_EQUAL(manifold.get_p(), 3);
    EXPECT(assert_equal(manifold.matrix(), mat));
}

// Test dynamic type constructor from fixed type
TEST(StiefelManifoldTest, FixedToDynamicConstructor) {
    StiefelManifold<2, 3> fixed_R;
    StiefelManifoldKP dynamic_R(fixed_R);
    Eigen::MatrixXd fixed_matrix  =fixed_R.matrix();
    Eigen::MatrixXd dynamic_matrix  =dynamic_R.matrix();
    CHECK(fixed_matrix.isApprox(dynamic_matrix, 1e-8));
}

// Test random sample
TEST(StiefelManifoldTest, RandomSample) {
    StiefelManifold<2, 3> manifold;
    auto randomSample = manifold.random_sample();
    EXPECT_LONGS_EQUAL(randomSample.get_k(), 2);
    EXPECT_LONGS_EQUAL(randomSample.get_p(), 3);
    EXPECT_LONGS_EQUAL(randomSample.matrix().rows(), 3);
    EXPECT_LONGS_EQUAL(randomSample.matrix().cols(), 2);
}

//Test projection from matrix to manifold
TEST(StiefelManifoldTest, ProjectToManifold) {
    Eigen::MatrixXd mat(3, 2);
    mat << 1, 2,
            3, 4,
            5, 6;
    auto projected = StiefelManifold<2, 3>::projectToManifold(mat);
    EXPECT_LONGS_EQUAL(projected.get_k(), 2);
    EXPECT_LONGS_EQUAL(projected.get_p(), 3);
    EXPECT_LONGS_EQUAL(projected.matrix().rows(), 3);
    EXPECT_LONGS_EQUAL(projected.matrix().cols(), 2);

    // Verify orthonormality (orthogonal constraint of manifold)
    Eigen::MatrixXd identity = projected.matrix().transpose() * projected.matrix();
    CHECK(identity.isApprox(Eigen::MatrixXd::Identity(2, 2), 1e-6));
}

// Test retraction
TEST(StiefelManifoldTest, RetractMatrix) {
    Eigen::MatrixXd mat(3, 2);
    mat << 1, 2,
            3, 4,
            5, 6;
    auto manifold = StiefelManifoldKP::projectToManifold(mat);
    Eigen::MatrixXd tangent(3, 2);
    tangent << 0.1, 0.2,
            0.3, 0.4,
            0.5, 0.6;
    auto retracted = manifold.retractMatrix(tangent);
    EXPECT_LONGS_EQUAL(retracted.get_k(), 2);
    EXPECT_LONGS_EQUAL(retracted.get_p(), 3);

    // Verify orthonormality (orthogonal constraint of manifold)
    Eigen::MatrixXd identity = retracted.matrix().transpose() * retracted.matrix();
    CHECK(identity.isApprox(Eigen::MatrixXd::Identity(2, 2), 1e-6));
}

// Test SymBlockDiagProduct
TEST(StiefelManifoldTest, SymBlockDiagProduct) {
    Eigen::MatrixXd A(3, 2);
    A << 1, 2,
            3, 4,
            5, 6;
    Eigen::MatrixXd B(3, 2);
    B << 1, 0,
            0, 1,
            0, 0;
    Eigen::MatrixXd C(3, 2);
    C << 2, 0,
            0, 2,
            0, 0;
    auto manifold = StiefelManifoldKP::projectToManifold(A);
    auto result = manifold.SymBlockDiagProduct(A, B.transpose(), C);
    // Test dimension
    EXPECT_LONGS_EQUAL(result.rows(), 3);
    EXPECT_LONGS_EQUAL(result.cols(), 2);
}

// Test projection to tangent space
TEST(StiefelManifoldTest, ProjectToTangentSpace) {
    Eigen::MatrixXd V(3, 2);
    V << 0.1, 0.2,
            0.3, 0.4,
            0.5, 0.6;
    StiefelManifold<2, 3> manifold;
    auto tangentProjection = manifold.projectToTangentSpace(V);
    // The "outer" dimension of matrix in tangent space is the same as corresponding matrix in manifold
    EXPECT_LONGS_EQUAL(tangentProjection.rows(), 3);
    EXPECT_LONGS_EQUAL(tangentProjection.cols(), 2);
}

// Test equality with tolerance
TEST(StiefelManifoldTest, Equals) {
    Eigen::MatrixXd mat1(3, 2);
    mat1 << 1, 0,
            0, 1,
            0, 0;
    Eigen::MatrixXd mat2(3, 2);
    mat2 << 1.00001, 0,
            0, 0.99999,
            0, 0;
    auto manifold1 = StiefelManifoldKP::projectToManifold(mat1);
    auto manifold2 = StiefelManifoldKP::projectToManifold(mat2);
    CHECK(manifold1.equals(manifold2, 1e-4));
}

// Test vectorize and devectorize
TEST(StiefelManifoldTest, VectorizeDeVectorizeConsistency) {
    // Define dimensions for the test
    constexpr int K = 2;
    constexpr int P = 3;

    // Create a sample StiefelManifold object
    StiefelManifold<K, P> manifold;

    // Create a test matrix
    Eigen::MatrixXd testMatrix(3, 3);
    testMatrix << 1, 2, 3,
            3, 4, 3,
            5, 6, 3;

    // Vectorize the test matrix
    auto vectorized = manifold.Vectorize(testMatrix);

    // Check if the vectorized result has the expected size
    EXPECT_LONGS_EQUAL(vectorized.size(), testMatrix.rows() * testMatrix.cols());

    // Devectorize back to the matrix
    auto devectorized = manifold.DeVectorize(vectorized, testMatrix.rows(), testMatrix.cols());

    // Check if the devectorized matrix is the same as the original
    EXPECT_LONGS_EQUAL(devectorized.rows(), testMatrix.rows());
    EXPECT_LONGS_EQUAL(devectorized.cols(), testMatrix.cols());
    CHECK(devectorized.isApprox(testMatrix, 1e-8));
}

// Test retraction
TEST(StiefelManifoldTest, RetractVectorize)
{
    Eigen::MatrixXd mat(3, 2);
    mat << 1, 2,
            3, 4,
            5, 6;
    auto manifold = StiefelManifold<2, 3>::projectToManifold(mat);
    CHECK(TangentSpace::isOnStiefelManifold(manifold.matrix()));
    TangentSpace tangentSpaceSampled(manifold.matrix());
    Matrix  randomTangentVector = tangentSpaceSampled.randomTangentVector();

    // Test whether the tangent vector satisfy the constraint of tangent space:
    // i.e. Y.transpose() * basis[i] + basis[i].transpose() * Y = 0
    EXPECT(tangentSpaceSampled.isInTangentSpace((randomTangentVector)));
    // The result of retract should be an element on the Stiefel manifold
    // i.e. satisfying the constraints.
    EXPECT(TangentSpace::isOnStiefelManifold(manifold.retractMatrix(1e-2 * randomTangentVector).matrix()));
    // Numerical verification, based on Definition 3.47 of Nicolas Boumal's "An introduction to optimization on smooth manifolds"
    // Rx(t * v) - x = t * v， given a small step size of t = 1e-4
    double F_norm =  (manifold.retractMatrix(1e-4 * randomTangentVector).matrix() - manifold.matrix() - 1e-4 * randomTangentVector).norm();
    EXPECT_LONGS_EQUAL(F_norm, 1e-6);
}

// Check constructor 
TEST(StiefelManifoldTest, CheckTangentSpaceConstructor)
{
    Eigen::MatrixXd Id_in = Eigen::MatrixXd::Identity(3,3);
    TangentSpace x(Id_in);
    EXPECT_LONGS_EQUAL(x.dim, 3); // Check correct dimension
    EXPECT(assert_equal(x.Y, Id_in)); // Check proper matrix assignment
}

// Test basis of the tangent space
TEST(StiefelManifoldTest, CheckSkewBasis)
{
    Eigen::MatrixXd Id_in = Eigen::MatrixXd::Random(3,2);
    auto M = StiefelManifold<2, 3>::projectToManifold(Id_in);
    EXPECT(TangentSpace::isOnStiefelManifold(M.matrix()));
    TangentSpace x(M.matrix());
    std::vector<Eigen::MatrixXd> skewBasis = x.generateSkewBasis();

    // Verify symmetric
    for(auto i = 0; i < skewBasis.size(); i++)
    {
        CHECK((skewBasis[i].transpose()).isApprox(-skewBasis[i], 1e-6));
    }

    // Verify orthogonality of orthogonal complement
    auto Q = x.generateXPerp();
    EXPECT_LONGS_EQUAL((x.Y.transpose() * Q).norm(), 1e-8);

    // Verify the orthogonality of basis
    for (auto i = 0; i < x.basis.size(); i++)
    {
        EXPECT(x.isInTangentSpace(x.basis[i]))
        EXPECT(x.basis[i].norm() == 1);
        for (auto j = 0; j < x.basis.size(); j++) {
            if (i!=j)
            EXPECT(((Eigen::Map<const Matrix>(x.basis[i].data(), M.dim(), 1)).transpose()
            * (Eigen::Map<const Matrix>(x.basis[j].data(), M.dim(), 1))).norm() == 0);
        }
    }

    // Test tangent matrix of from random tangent vector
    Eigen::MatrixXd randTVec = x.randomTangentVector();
    x.verifyTangentConstraints();
    EXPECT(x.isInTangentSpace(randTVec));
}

// Test for Lift Method
TEST(StiefelManifoldTest, TestLiftMethod) {
    // Define dimensions for the smaller and larger manifold
    constexpr int K = 2;  // Columns
    constexpr int P = 2;  // Rows in the small manifold
    constexpr int P_LIFT = 6;  // Rows in the larger manifold

    // Create a small orthonormal matrix (2x2)
    Eigen::Matrix<double, P, K> R;
    R << 1, 0,
            0, 1;

    // Use the Lift method to create a larger manifold (4x2)
    auto lifted = StiefelManifoldKP::Lift(P_LIFT, R);
    EXPECT(TangentSpace::isOnStiefelManifold(lifted.matrix()));

    std::cout << "lifted : " << lifted.matrix() << std::endl;
    // Validate the dimensions of the lifted matrix
    EXPECT_LONGS_EQUAL(lifted.matrix().rows(), P_LIFT);
    EXPECT_LONGS_EQUAL(lifted.matrix().cols(), K);

    // Check that the top-left block of the lifted matrix matches R
    Eigen::Matrix<double, P, K> topLeft = lifted.matrix().topLeftCorner(P, K);
    CHECK(topLeft.isApprox(R, 1e-10));

    // Ensure that the rest of the lifted matrix is zero
    Eigen::MatrixXd bottomPart = lifted.matrix().bottomRows(P_LIFT - P);
    CHECK(bottomPart.isZero(1e-10));

    // Validate orthonormality of the top-left block
    Eigen::Matrix<double, K, K> identityCheck = topLeft.transpose() * topLeft;
    CHECK(identityCheck.isApprox(Eigen::Matrix<double, K, K>::Identity(), 1e-10));

    // Check the orthonormality of lifted stiefel manifold element
    CHECK((lifted.matrix().transpose() * lifted.matrix()).isApprox(Matrix::Identity(K, K), 1e-8));
}


/* ************************************************************************* */
int main() {
    TestResult tr;
    return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
