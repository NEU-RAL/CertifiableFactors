/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include "../StiefelManifold.h"
#include "../SEsyncFactor.h"
#include "../StiefelManifoldPriorFactor.h"
#include "../LiftedPosePriorFactor.h"
#include <iostream>
#include <gtsam/base/Matrix.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/testLie.h>
#include <gtsam/geometry/SO4.h>
#include <gtsam/geometry/Pose2.h>

using namespace gtsam;

// Test Jacobian with GTSAM's numerical Jacobian TEST MACRO for StiefelManifold prior factor, d = 2
TEST(StiefelManifoldPriorFactorTest, StiefelNumericalJacobain2) {
    size_t d = 2;
    const Rot2 R1(0.3);
    const Rot2 R1_initial(0.301);

    for (const int p : {5, 4, 3, 2}) {
        auto model = noiseModel::Isotropic::Sigma(p * d, 1.2);
        Matrix M = Matrix::Zero(p, d);
        M.block(0, 0, 2, 2) = R1.matrix();
        StiefelManifoldKP Y1(M);
        Y1.initializeTangentSpace();
        M.block(0, 0, 2, 2) = R1_initial.matrix();
        StiefelManifoldKP Y2(M);
        Y2.initializeTangentSpace();
        auto factor = StPriorFactor2(1, Y2, p, model);
        Matrix H1;
        factor.evaluateError(Y1, H1);
        Values values;
        values.insert(1, Y1);
        EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-7, 1e-5);
    }
}

// Test Jacobian with GTSAM's numerical Jacobian TEST MACRO for LiftedPose prior factor, d = 2
TEST(LiftedPosePriorFactorTest, LiftedPoseNumericalJacobain2) {
    size_t d = 2;
    const Rot2 R1(0.3);
    const Rot2 R1_initial(0.301);
    Vector t1(d), t1_initial(d);
    t1 << 0.1, 0.3;
    t1_initial << 0.101, 0.302;
    for (const int p : {5, 4, 3, 2}) {
        auto model = noiseModel::Isotropic::Sigma(p * d + p, 1.2);
        Matrix M = Matrix::Zero(p, d);
        M.block(0, 0, 2, 2) = R1.matrix();
        StiefelManifoldKP Y1(M);
        Y1.initializeTangentSpace();
        M.block(0, 0, 2, 2) = R1_initial.matrix();
        StiefelManifoldKP Y2(M);
        Y2.initializeTangentSpace();
        LiftedPoseDP P1(Y1, LiftedPoseDP::LiftToRp(t1, p));
        LiftedPoseDP P2(Y2, LiftedPoseDP::LiftToRp(t1_initial, p));
        auto factor = LiftedPosePriorFactor2 (1, P2, p, model);
        Matrix H1;
        factor.evaluateError(P1, H1);
        Values values;
        values.insert(1, P1);
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

// Test Jacobian with GTSAM's numerical Jacobian TEST MACRO for StiefelManifold prior factor, d = 3
TEST(StiefelManifoldPriorFactorTest, StiefelNumericalJacobain3) {
    size_t d = 3;
    for (const size_t p : {5, 4, 3}) {
        auto model = noiseModel::Isotropic::Sigma(p * d, 1.2);
        Matrix M = Matrix::Zero(p, d);
        M.topLeftCorner(3, 3) = submanifold::R1.matrix();
        StiefelManifoldKP Q1(M);
        Q1.initializeTangentSpace();
        M.topLeftCorner(3, 3) = submanifold::R2.matrix();
        StiefelManifoldKP Q2(M);
        Q2.initializeTangentSpace();
        auto factor = StPriorFactor3 (1, Q2, p, model);
        Matrix H1;
        factor.evaluateError(Q1, H1);
        Values values;
        values.insert(1, Q1);
        EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-7, 1e-5);
    }
}

// Test Jacobian with GTSAM's numerical Jacobian TEST MACRO for LiftedPose prior factor, d = 3
TEST(LiftedPosePriorFactorTest, LiftedPoseNumericalJacobain3) {
    size_t d = 3;
    Vector t1(3), t2(3), t12(3);
    t1 << 0.3, 0.2, 0.1;
    t12 << 0.001, 0.001, 0.001;
    t2 = submanifold::R2.matrix().inverse() * t12;
    for (const size_t p : {5, 4, 3}) {
        auto model = noiseModel::Isotropic::Sigma(p * d + p, 1.2);
        Matrix M = Matrix::Zero(p, d);
        M.topLeftCorner(3, 3) = submanifold::R1.matrix();
        StiefelManifoldKP Q1(M);
        Q1.initializeTangentSpace();
        M.topLeftCorner(3, 3) = submanifold::R2.matrix();
        StiefelManifoldKP Q2(M);
        Q2.initializeTangentSpace();

        LiftedPoseDP P1(Q1, LiftedPoseDP::LiftToRp(t1, p));
        LiftedPoseDP P2(Q2, LiftedPoseDP::LiftToRp(t2, p));
        auto factor = LiftedPosePriorFactor3 (1, P2, p, model);
        Matrix H1;
        factor.evaluateError(P1, H1);
        Values values;
        values.insert(1, P1);
        EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-7, 1e-5);
    }
}

/* ************************************************************************* */
int main() {
    TestResult tr;
    return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
