/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */
#include "../CertifiableRA.h"

using namespace std;
using namespace gtsam;

/**
 * @brief Entry point for the certifiable Rotation Averaging solver.
 *
 * Parses command‑line arguments to configure the ambient dimension (d), lifted
 * dimension (p), input dataset path, and output file path. Loads measurements
 * from a G2O file, initializes and solves the certifiable Rotation Averaging
 * problem for 2D or 3D, exports the optimized rotations, and writes the
 * certificate results to a CSV.
 *
 * @param argc Number of command‑line arguments.
 * @param argv Array of C‑string arguments:
 *             argv[1] = d (ambient dimension: 2 or 3)
 *             argv[2] = p (lifted dimension)
 *             argv[3] = dataset file path (.g2o)
 *             argv[4] = output file path
 * @return Zero on success; throws on error.
 */
int main(int argc, char* argv[]) {

    // Input and output file paths
    string inputFile, outputFile;

    // Parse ambient dimension and lifted dimension
    int d = stoi(argv[1]);
    int p = stoi(argv[2]);
    inputFile  = string(argv[3]);
    outputFile = string(argv[4]);

    // Range of lifted dimensions to explore
    size_t pMin = p;
    size_t pMax = d + 10;

    // Number of poses loaded from the dataset
    size_t num_poses;

    // Container for the certificate results
    optional<CertificateResults> result;

    // Load measurements from the G2O file
    auto measurements = DataParser::read_g2o_file(inputFile, num_poses);
    std::cout << "Loaded " << measurements.poseMeasurements.size() << " measurements between "
              << num_poses << " poses from file "  << inputFile << std::endl;

    if (d == 2) {
        // Initialize 2D certifiable Rotation Averaging problem
        auto RA_problem = std::make_shared<CertifiableRA2>(p, measurements);
        RA_problem->init();

        // Validate that the minimum lifted dimension is not below the ambient
        if (pMin < d) {
            throw std::runtime_error("pMin is smaller than the base dimension d");
        }

        // Solve over the specified range and export the rounded solution
        result = RA_problem->Solve(pMin, pMax);
        auto elementMatrix = RA_problem->RoundSolutionS();
        RA_problem->ExportData(outputFile,elementMatrix,true);
    }
    else if (d == 3) {
        // Initialize 3D certifiable Rotation Averaging problem
        auto RA_problem = std::make_shared<CertifiableRA3>(p, measurements);
        RA_problem->init();

        // Validate that the minimum lifted dimension is not below the ambient
        if (pMin < d) {
            throw std::runtime_error("pMin is smaller than the base dimension d");
        }

        // Solve over the specified range and export the rounded solution
        result = RA_problem->Solve(pMin, pMax);
        auto elementMatrix = RA_problem->RoundSolutionS();
        RA_problem->ExportData(outputFile,elementMatrix,true);
    }
    // Export certificate results to CSV
    CertificateResults::exportCertificateResultsSingleCSV(result.value(),outputFile);
}
