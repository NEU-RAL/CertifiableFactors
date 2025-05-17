//
// Created by nikolas on 5/2/25.
//

#ifndef CERTIFIABLERESULTS_H
#define CERTIFIABLERESULTS_H

#pragma once

#include "types.h"
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <Eigen/CholmodSupport>
#include <Eigen/Geometry>
#include <Eigen/SPQRSupport>

typedef double Scalar;
typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;

namespace gtsam
{

    /**
     * @brief Container for results produced by the certifiable optimization routines.
     *
     * Stores the relaxation ranks, optimized variables, certificate values, timing metrics,
     * and methods for exporting results.
     */
    class CertificateResults
    {
    public:
        /** @brief Relaxation rank at which optimization was initialized. */
        size_t startingRank;

        /** @brief Relaxation rank at which solution was certified. */
        size_t endingRank;

        /** @brief Variable matrix Y at the certified solution (stacked manifold representation). */
        SparseMatrix Yopt;

        /**
         * @brief Objective values F(Z) = F(YᵀY) recorded at each relaxation level.
         *
         * Z is the Gram matrix of Y.
         */
        std::vector<Scalar> SDPval;

        /**
         * @brief Norms of the Riemannian gradient ∥∇F(Y)∥ at each relaxation level.
         */
        std::vector<Scalar> gradnorm;

        /**
         * @brief Dual Lagrange multiplier matrix Λ corresponding to the certified Yopt.
         *
         * Computed per eq. (119) of the SE‑Sync technical report.  If Z = YᵀY
         * solves the dual SDP exactly, then Λ solves the corresponding primal
         * Lagrangian relaxation.
         */
        SparseMatrix Lambda;

        /**
         * @brief Rounded solution x̂ = [t | R] in SE(d)ⁿ after projecting onto the manifold.
         */
        Matrix xhat;

        /** @brief Total wall‑clock time (ms) for the entire certifiable algorithm. */
        double total_computation_time;

        /**
         * @brief Initialization times (ms) for each rank’s random or descent-based start.
         */
        std::vector<double> initialization_time;

        /** @brief Optimization times (ms) taken by Levenberg–Marquardt at each level. */
        std::vector<double> elapsed_optimization_times;

        /**
         * @brief Verification times (ms) spent on eigenvalue tests per relaxation level.
         */
        std::vector<double> verification_times;


        /**
         * @brief Export all fields of CertificateResults to a single CSV file.
         *
         * Writes a four‑column CSV with rows labeled by field name,
         * index within vector fields, and the corresponding value.
         *
         * @param R         CertificateResults instance to export.
         * @param filename  Path to the output CSV file.
         */
        static void exportCertificateResultsSingleCSV(
            const CertificateResults& R,
            const std::string& filename)
        {
            std::ofstream out(filename);
            out << "field,i,j,value\n";

            // 1) Export SDPval vector entries
            for (size_t k = 0; k < R.SDPval.size(); ++k) {
                out << "SDPval," << k << ",," << R.SDPval[k] << "\n";
            }

            // 2) Export gradnorm vector entries
            for (size_t k = 0; k < R.gradnorm.size(); ++k) {
                out << "gradnorm," << k << ",," << R.gradnorm[k] << "\n";
            }

            // 3) Export scalar fields
            out << "total_computation_time,,," << R.total_computation_time << "\n";
            out << "startingRank,,,"         << R.startingRank         << "\n";
            out << "endingRank,,,"           << R.endingRank           << "\n";

            // 4) Export initialization_time vector entries
            for (size_t k = 0; k < R.initialization_time.size(); ++k) {
                out << "initialization_time," << k << ",," << R.initialization_time[k] << "\n";
            }

            // 5) Export elapsed_optimization_times vector entries
            for (size_t k = 0; k < R.elapsed_optimization_times.size(); ++k) {
                out << "elapsed_optimization_times," << k << ",,"
                    << R.elapsed_optimization_times[k] << "\n";
            }

            // 6) Export verification_times vector entries
            for (size_t k = 0; k < R.verification_times.size(); ++k) {
                out << "verification_times," << k << ",,"
                    << R.verification_times[k] << "\n";
            }

            std::cout << "All fields exported to \"" << filename << "\"\n";
        }
    };

}  // namespace gtsam




#endif //CERTIFIABLERESULTS_H
