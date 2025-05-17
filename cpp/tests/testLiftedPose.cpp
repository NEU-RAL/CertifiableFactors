/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include "../LiftedPose.h"
#include <iostream>
#include <gtsam/base/Matrix.h>
#include <CppUnitLite/TestHarness.h>
#include <gtsam/geometry/SO4.h>

using namespace gtsam;

// Test for LiftedPose constructor and basic getters methods.
TEST(LiftedPoseTest, ConstructorAndBasicGetters) {
    constexpr int D = 3; // Dimensions for Stiefel element
    constexpr int P = 5; // Lifted pose rows

    // Create Stiefel element and translation vector
    StiefelManifoldKP Y(StiefelManifoldKP::Random(std::default_random_engine::default_seed, D, P));
    Vector t_D = Vector::Random(D);
    // Lifted translation from D to P
    Vector t_P = LiftedPoseDP::LiftToRp(t_D, P);
    // Create LiftedPose
    LiftedPoseDP pose(Y, t_P);

    // Check matrix dimensions
    EXPECT_LONGS_EQUAL(pose.get_Rows(), P);
    EXPECT_LONGS_EQUAL(pose.get_Cols(), D + 1);

    // Check Stiefel element and translation vector
    EXPECT(pose.get_StiefelElement().matrix().isApprox(Y.matrix()));
    EXPECT(pose.get_TranslationVector().isApprox(t_P));

    // Check lifted translation vector
    Vector liftedT = pose.get_LiftedTranslationVector();
    EXPECT_LONGS_EQUAL(liftedT.size(), P);
    EXPECT(liftedT.head(D).isApprox(t_D));
}

// Test Lift and project of translation vectors
TEST(LiftedPoseTest, LiftAndProjectTranslationVector) {
    constexpr int D = 3; // Dimensions for Stiefel element
    constexpr int P = 5; // Lifted pose rows

    // Generate a random translation vector
    Vector t = Vector::Random(D);

    // Test LiftToRp
    Vector liftedT = LiftedPoseDP::LiftToRp(t, P);
    EXPECT_LONGS_EQUAL(liftedT.size(), P);
    EXPECT(liftedT.head(D).isApprox(t));
    EXPECT((liftedT.tail(P - D).array() == 0).all());

    // Test ProjectToRd
    Vector projectedT = LiftedPoseDP::ProjectToRd(liftedT, D);
    EXPECT_LONGS_EQUAL(projectedT.size(), D);
    EXPECT(projectedT.isApprox(t));
}

// Test random sample of LiftedPose
TEST(LiftedPoseTest, RandomSample) {
    constexpr int D = 3;
    constexpr int P = 5;

    // Create a random LiftedPose sample
    auto randomPose = LiftedPoseDP::Random(std::default_random_engine::default_seed, D, P);

    // Check the dimensions of the random pose
    EXPECT_LONGS_EQUAL(randomPose.get_Rows(), P);
    EXPECT_LONGS_EQUAL(randomPose.get_Cols(), D + 1);

    // Check that the lifted translation vector matches the size
    EXPECT_LONGS_EQUAL(randomPose.get_LiftedTranslationVector().size(), P);
}

// Test retraction
// LiftedPose is composed by StiefelManifold and a Vector, so the retraction is just putting Stiefel's and Vector"s(just addition) together.
// Only check the dimension, the numerical check is done in Unit test of Stiefel manifold(Potentially add numerical check later)
TEST(LiftedPoseTest, Retraction) {
    constexpr int D = 3;
    constexpr int P = 5;

    // Create initial LiftedPose
    StiefelManifoldKP Y(StiefelManifoldKP::Random(std::default_random_engine::default_seed, D, P));
    Vector t = Vector::Random(D);
    Vector t_lifted = LiftedPoseDP::LiftToRp(t, P);
    LiftedPoseDP pose(Y, t_lifted);

    // Create a random tangent vector
    size_t dim = pose.dim();
    auto V = LiftedPoseDP::TangentVector::Random(dim);

    // Perform retraction
    LiftedPoseDP retractedPose = pose.retract(V);

    // Check dimensions of retracted pose
    EXPECT_LONGS_EQUAL(retractedPose.get_Rows(), P);
    EXPECT_LONGS_EQUAL(retractedPose.get_Cols(), D + 1);

}

// Test equals methods
TEST(LiftedPoseTest, equals) {
    constexpr int D = 3;
    constexpr int P = 5;

    // Create LiftedPose
    StiefelManifoldKP Y(StiefelManifoldKP::Random(std::default_random_engine::default_seed, D, P));
    Vector t = Vector::Random(D);
    Vector t_lifted = LiftedPoseDP::LiftToRp(t, P);
    LiftedPoseDP pose1(Y, t_lifted);
    LiftedPoseDP pose2(Y, t_lifted);

    // Check equality
    EXPECT(pose1.equals(pose2, 1e-6));

    // Modify pose2 and ensure it's no longer equal
    Vector tModified = t + Vector::Ones(D) * 10;
    Vector tModified_lifted = LiftedPoseDP::LiftToRp(tModified, P);
    LiftedPoseDP pose3(Y, tModified_lifted);
    EXPECT(!(pose1.equals(pose3, 1e-6)));
}

/* ************************************************************************* */
int main() {
    TestResult tr;
    return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
