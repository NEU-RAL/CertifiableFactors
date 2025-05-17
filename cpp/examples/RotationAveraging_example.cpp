//
// Created by Nikolas on 2/13/25.
//

#include <gtsam/base/timing.h>
#include <gtsam/slam/InitializePose.h>
#include "../RaFactor.h"
#include "../LiftedPose.h"
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include "../utils.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace gtsam;

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

    for (const auto & meas : measurements) {
        double Kappa = meas.kappa;
        Vector sigmas(p*d);
        sigmas.setConstant(sqrt(1/ ( 2 * Kappa)));

        noiseModel::Diagonal::shared_ptr noise = noiseModel::Diagonal::Sigmas(sigmas);
        if (d == 2) {
            inputGraph.emplace_shared<RaFactor2>(meas.i, meas.j, Rot2::atan2(meas.R(1,0),meas.R(0,0)), p, noise);

        }
        else if (d == 3 ) {
            inputGraph.emplace_shared<RaFactor3>(meas.i, meas.j, Rot3(meas.R), p, noise);

        }
        else {
            std::cerr << "Un" << std::endl;
        }
    }

    Values initial;
    for (size_t j = 0; j < num_poses; j++) {
        StiefelManifoldKP Y = StiefelManifoldKP::Random(std::default_random_engine::default_seed, d, p);
        initial.insert(j, Y);
    }


    Values::shared_ptr posesInFile;
    Values poses;
    auto lmParams = LevenbergMarquardtParams::CeresDefaults();

    lmParams.maxIterations = 1000;
    lmParams.relativeErrorTol = 1e-20;
    lmParams.verbosityLM = LevenbergMarquardtParams::SUMMARY;

    auto lm = std::make_shared<LevenbergMarquardtOptimizer>(inputGraph, initial, lmParams);
    auto results = lm->optimize();

    return 0;
}
