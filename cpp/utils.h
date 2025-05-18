/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson, David M. Rosen.
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */
#ifndef STIEFELMANIFOLDEXAMPLE_UTILS_H
#define STIEFELMANIFOLDEXAMPLE_UTILS_H

#include <string>
#include <Eigen/Sparse>
#include "RelativePoseMeasurement.h"
#include "types.h"

namespace gtsam {
    namespace DataParser {

/** Given the name of a file containing a description of a special Euclidean
 * synchronization problem expressed in the .g2o format (i.e. using "EDGE_SE2 or
 * EDGE_SE3:QUAT" measurements), this function constructs and returns the
 * corresponding vector of RelativePoseMeasurements, and reports the total
 * number of poses in the pose-graph */
        Measurement read_g2o_file(const std::string &filename, size_t &num_poses);


        Measurement read_pycfg_file(const std::string &filename);


        /** Given two matrices X, Y in SO(d)^n, this function computes and returns the
 * orbit distance d_S(X,Y) between them and (optionally) the optimal
 * registration G_S in SO(d) aligning Y to X, as described in Appendix C.1 of
 * the SE-Sync tech report.
 */
        Scalar dS(const Matrix &X, const Matrix &Y, Matrix *G_S = nullptr);

/** Given two matrices X, Y in O(d)^n, this function computes and returns the
 * orbit distance d_O(X,Y) between them and (optionally) the optimal
 * registration G_O in O(d) aligning Y to X, as described in Appendix C.1 of the
 * SE-Sync tech report.
 */
        Scalar dO(const Matrix &X, const Matrix &Y, Matrix *G_O = nullptr);

    } // namespace DataParser
} // namespace gtsam




#endif //STIEFELMANIFOLDEXAMPLE_UTILS_H
