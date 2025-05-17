//
// Created by jason on 1/30/25.
//
#include <iomanip>
#include <gtsam/base/timing.h>
//#include <gtsam/sfm/ShonanAveraging.h>
#include <gtsam/slam/InitializePose.h>
#include <gtsam/slam/dataset.h>
#include "../utils.h"
#include "../RaFactor.h"
#include "../LiftedPose.h"
#include "../SEsyncFactor.h"
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace gtsam;

LiftedPoseDP LiftedToP(const Pose3 &pose3_, const size_t p) {
    return LiftedPoseDP(StiefelManifoldKP::Lift(p, pose3_.rotation().matrix()), pose3_.translation());
}

int main(int argc, char* argv[]) {

    /*
    * argv[1] = d
    * argv[2] = p
    * argv[3] = dataset
    * argv[4] = output file
    *
    * */

    std::string inputFile;
    std::string outputFile;

    // Parse spatial dimension (d) and lifted dimension (p) from arguments
    int d = std::stoi(argv[1]);
    int p = std::stoi(argv[2]);

    inputFile  = std::string(argv[3]);  // Path to the input dataset
    outputFile = std::string(argv[4]);  // Path for writing output data

    // Range of lifted dimensions to test
    size_t pMin = p;
    size_t pMax = d + 10;

    // Number of poses in the dataset
    size_t num_poses;

    auto measurements = DataParser::read_g2o_file(inputFile, num_poses);
    std::cout << "Loaded " << measurements.poseMeasurements.size() << " measurements between "
              << num_poses << " poses from file "  << inputFile << std::endl;

    // Random generator from Dave's code
    std::default_random_engine generator(std::default_random_engine::default_seed);
    std::normal_distribution<double> g;

    NonlinearFactorGraph inputGraph;

    for (const auto & meas : measurements.poseMeasurements) {
        double Kappa = meas.kappa;
        double tau = meas.tau;
        Vector sigmas = Vector::Zero(p * d + p);
        sigmas.head(p * d).setConstant(sqrt(1/ ( 2 * Kappa)));
        sigmas.tail(p).setConstant( sqrt(1/  (2 * tau)));
        noiseModel::Diagonal::shared_ptr noise = noiseModel::Diagonal::Sigmas(sigmas);


        if (d == 2) {
            inputGraph.emplace_shared<SEsyncFactor2>(meas.i, meas.j, meas.R, meas.t, p, noise);

        }
        else if (d ==3 ) {
            inputGraph.emplace_shared<SEsyncFactor3>(meas.i, meas.j, meas.R, meas.t, p, noise);

        }
        else {
            std::cerr << "Un" << std::endl;
        }

    }

    NonlinearFactorGraph::shared_ptr graph;
    Values::shared_ptr initials;
    bool is3D = true;
    std::tie(graph, initials) = readG2o(inputFile, is3D);

    Values initials_g2o;
    for (size_t k = 0; k < initials->size(); k++) {
        Pose3 retrieved_pose;
        if (initials->exists(k))
            retrieved_pose = initials->at<gtsam::Pose3>(k);
        initials_g2o.insert(k, LiftedToP(retrieved_pose, p));
    }

    Values initial;
    for (size_t j = 0; j < num_poses; j++) {
        StiefelManifoldKP Y = StiefelManifoldKP::Random(std::default_random_engine::default_seed, d, p);

        Vector trans = Vector::Zero(p);
        for (int i = 0; i < p; i++) {
            trans(i) = g(generator);
        }
        initial.insert(j, LiftedPoseDP(Y, trans));
    }


    Values::shared_ptr posesInFile;
    Values poses;
    auto lmParams = LevenbergMarquardtParams::CeresDefaults();
//    LevenbergMarquardtParams lmParams;
    lmParams.maxIterations = 1000;
    lmParams.relativeErrorTol = 1e-5;
    lmParams.verbosityLM = LevenbergMarquardtParams::SUMMARY;

    auto lm = std::make_shared<LevenbergMarquardtOptimizer>(inputGraph, initial, lmParams);
    auto results = lm->optimize();



    return 0;
}

/* ************************************************************************* */
