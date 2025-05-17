/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include "../StiefelManifold.h"
#include "../StiefelManifold-inl.h"
#include "../RaFactor.h"
#include <iostream>
#include <gtsam/base/Matrix.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/testLie.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/SO4.h>

using namespace gtsam;

// Test initialization and printing
TEST(RaFactorTest, Initialization) {

    auto model = noiseModel::Isotropic::Sigma(3*2, 1.2);
    auto R12 = Rot2::fromAngle(M_PI / 4);
    RaFactor2 factor(1, 2, R12, 3, model);

    factor.print("RaFactor<2>: ");
}

// Test error evaluation
TEST(RaFactorTest, EvaluateError) {
    auto model = noiseModel::Isotropic::Sigma(3*2, 1.2);
    auto R12 = Rot2::fromAngle(M_PI / 4);
    size_t p = 3;
    auto factor1 = RaFactor2(1, 2, R12, p, model);
    const std::default_random_engine::result_type seed = std::default_random_engine::default_seed;

    // Mock data for Q1 and Q2
    StiefelManifoldKP Q1 = StiefelManifoldKP::Random(seed, 2, 3);
    Q1.initializeTangentSpace();
    StiefelManifoldKP Q2 = StiefelManifoldKP::Random(seed, 2, 3);
    Q2.initializeTangentSpace();

    Matrix H1, H2;
    Vector error = factor1.evaluateError(Q1, Q2, H1, H2);
    EXPECT_LONGS_EQUAL(H1.rows(), Q1.get_p() * Q1.get_k());
    EXPECT_LONGS_EQUAL(H2.rows(), Q2.get_p() * Q2.get_k());
}

// Test Jacobian with GTSAM's numerical Jacobian TEST MACRO, d = 2
TEST(RaFactorTest, NumericalJacobain2) {
    constexpr int d = 2;
    const Rot2 R1(0.3), R2(0.5), R12(0.2);
    // Test on a series of dimension
    for (const int p : {5, 4, 3, 2}) {
        auto model = noiseModel::Isotropic::Sigma(d*p, 1.2);
        Matrix M = Matrix::Zero(p, d);
        M.block(0, 0, 2, 2) = R1.matrix();
        StiefelManifoldKP Q1(M);
        Q1.initializeTangentSpace();
        M.block(0, 0, 2, 2) = R2.matrix();
        StiefelManifoldKP Q2(M);
        Q2.initializeTangentSpace();
        auto factor = RaFactor2 (1, 2, R12, p, model);
        Matrix H1, H2;
        factor.evaluateError(Q1, Q2, H1, H2);
        Values values;
        values.insert(1, Q1);
        values.insert(2, Q2);
        EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-7, 1e-5);
    }
}

// Using those rotation expression from unit test of SOn from GTSAM
namespace so3 {
    SO3 id;
    Vector3 v1 = (Vector(3) << 0.1, 0, 0).finished();
    SO3 R1 = SO3::Expmap(v1);
    Vector3 v2 = (Vector(3) << 0.01, 0.02, 0.03).finished();
    SO3 R2 = SO3::Expmap(v2);
    SO3 R12 = R1.between(R2);
} // namespace so3

namespace submanifold {
    SO4 id;
    Vector6 v1 = (Vector(6) << 0, 0, 0, 0.1, 0, 0).finished();
    SO3 R1 = SO3::Expmap(v1.tail<3>());
    SO4 Q1 = SO4::Expmap(v1);
    Vector6 v2 = (Vector(6) << 0, 0, 0, 0.01, 0.02, 0.03).finished();
    SO3 R2 = SO3::Expmap(v2.tail<3>());
    SO4 Q2 = SO4::Expmap(v2);
    SO3 R12 = R1.between(R2);
} // namespace submanifold

// Test Jacobian with GTSAM's numerical Jacobian TEST MACRO, d = 3
TEST(RaFactorTest, NumericalJacobain3) {
    size_t d = 3;
    for (const size_t p : {5, 4, 3}) {
        auto model = noiseModel::Isotropic::Sigma(d*p, 1.2);
        Matrix M = Matrix::Zero(p, d);
        M.topLeftCorner(3, 3) = submanifold::R1.matrix();
        StiefelManifoldKP Q1(M);
        Q1.initializeTangentSpace();
        M.topLeftCorner(3, 3) = submanifold::R2.matrix();
        StiefelManifoldKP Q2(M);
        Q2.initializeTangentSpace();
        auto factor = RaFactor3(1, 2, Rot3(::so3::R12.matrix()), p, model);
        Matrix H1, H2;
        factor.evaluateError(Q1, Q2, H1, H2);

        Values values;
        values.insert(1, Q1);
        values.insert(2, Q2);
        EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-7, 1e-5);
    }
}

/* ************************************************************************* */
int main() {
    TestResult tr;
    return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
