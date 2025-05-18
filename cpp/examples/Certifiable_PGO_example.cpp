/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */
#include "../CertifiablePGO.h"

using namespace std;
using namespace gtsam;

/**
 * @brief Entry point for the certifiable Pose Graph Optimization (PGO) solver.
 *
 * Parses command-line arguments to configure the spatial dimension and lifted-pose dimension,
 * loads measurement data from a G2O file, initializes and solves the Certifiable PGO problem
 * for 2D or 3D data, exports the optimized trajectory, and writes certificate results to CSV.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of C-string arguments:
 *             argv[1] = d (spatial dimension: 2 or 3)
 *             argv[2] = p (lifted-pose dimension)
 *             argv[3] = dataset file path (input .g2o file)
 *             argv[4] = output file path for exporting results
 * @return     0 on success, non-zero on failure.
 */
int main(int argc, char* argv[]) {

    // Input and output file paths
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

    // Container for certificate results (populated after solving)
    std::optional<CertificateResults> result;

    // Load measurements from the G2O file
    auto measurements = DataParser::read_g2o_file(inputFile, num_poses);
    std::cout << "Loaded " << measurements.poseMeasurements.size()
              << " measurements between " << num_poses
              << " poses from file " << inputFile << std::endl;

    if (d == 2) {
        // Initialize 2D certifiable PGO problem
        auto PGO_problem = std::make_shared<CertifiablePGO2>(p, measurements);
        PGO_problem->init();

        // Ensure pMin is not below the base dimension
        if (pMin < d) {
            throw std::runtime_error("pMin is smaller than the base dimension d");
        }

        // Solve over the specified range of lifted dimensions
        result = PGO_problem->Solve(pMin, pMax);

        // Round and export the optimized solution
        auto elementMatrix = PGO_problem->RoundSolutionS();
        PGO_problem->ExportData(outputFile, elementMatrix, true);
    } else if (d == 3) {
        // Initialize 3D certifiable PGO problem
        auto PGO_problem = std::make_shared<CertifiablePGO3>(p, measurements);
        PGO_problem->init();

        if (pMin < d) {
            throw std::runtime_error("pMin is smaller than the base dimension d");
        }

        // Solve over the specified range of lifted dimensions
        result = PGO_problem->Solve(pMin, pMax);

        // Round and export the optimized solution
        auto elementMatrix = PGO_problem->RoundSolutionS();
        PGO_problem->ExportData(outputFile, elementMatrix, true);
    }

    // Export certificate results to a single CSV file
    CertificateResults::exportCertificateResultsSingleCSV(result.value(), outputFile);

    return 0;
}
