/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Geometry>
#include "utils.h"
#include <unordered_set>

namespace gtsam {
    namespace DataParser {
        Matrix fromAngle(double angle_rad) {
            Matrix rotation_matrix_2d(2, 2);
            rotation_matrix_2d << cos(angle_rad), -sin(angle_rad), sin(angle_rad),
                    cos(angle_rad);
            return rotation_matrix_2d;
        }

        Matrix fromQuat(double qx, double qy, double qz, double qw) {
            Eigen::Quaterniond q(qw, qx, qy, qz);
            auto rot_mat = q.toRotationMatrix();
            // Not sure why we can't cast it directly?
            Matrix result(3, 3);
            result << rot_mat(0, 0), rot_mat(0, 1), rot_mat(0, 2), rot_mat(1, 0),
                    rot_mat(1, 1), rot_mat(1, 2), rot_mat(2, 0), rot_mat(2, 1), rot_mat(2, 2);
            return result;
        }

        Measurement read_g2o_file(const std::string &filename, size_t &num_poses_) {
            std::unordered_set<size_t> pose_ids, landmark_ids;

            // Preallocate output vector
            Measurement measurements;
            RelativePoseMeasurement posemeasurement;
            RelativeLandmarkMeasurement landmarkmeasurement;


            // A string used to contain the contents of a single line
            std::string line;

            // A string used to extract tokens from each line one-by-one
            std::string token;

            // Preallocate various useful quantities
            Scalar dx, dy, dz, dtheta, dqx, dqy, dqz, dqw, I11, I12, I13, I14, I15, I16,
                    I22, I23, I24, I25, I26, I33, I34, I35, I36, I44, I45, I46, I55, I56, I66;

            size_t i, j;

            // Open the file for reading
            std::ifstream infile(filename);
            size_t &num_poses = measurements.num_poses;
            size_t &num_landmarks = measurements.num_landmarks;

            std::unordered_map<size_t, size_t> poses;
            std::unordered_map<size_t, size_t> landmarks;

            while (std::getline(infile, line)) {
                // Construct a stream from the string
                std::stringstream strstrm(line);

                // Extract the first token from the string
                strstrm >> token;

                if (token == "EDGE_SE2") {
                    // This is a 2D pose measurement

                    /** The g2o format specifies a 2D relative pose measurement in the
                     * following form:
                     *
                     * EDGE_SE2 id1 id2 dx dy dtheta, I11, I12, I13, I22, I23, I33
                     *
                     */

                    // Extract formatted output
                    strstrm >> i >> j >> dx >> dy >> dtheta >> I11 >> I12 >> I13 >> I22 >>
                            I23 >> I33;
                    if (poses.insert({i, num_poses}).second) num_poses++;

                    if (poses.insert({j, num_poses}).second) num_poses++;

                    // Pose ids
                    posemeasurement.i = poses[i];
                    posemeasurement.j = poses[j];
                    pose_ids.insert(i);
                    pose_ids.insert(j);

                    // Raw measurements
                    posemeasurement.t = Eigen::Matrix<Scalar, 2, 1>(dx, dy);
                    posemeasurement.R = Eigen::Rotation2D<Scalar>(dtheta).toRotationMatrix();

                    Eigen::Matrix<Scalar, 2, 2> TranInfo;
                    TranInfo << I11, I12, I12, I22;
                    posemeasurement.tau = 2 / TranInfo.inverse().trace();

                    posemeasurement.kappa = I33;



                    measurements.poseMeasurements.push_back(posemeasurement);

                } else if (token == "EDGE_SE3:QUAT") {

                    // This is a 3D pose measurement

                    /** The g2o format specifies a 3D relative pose measurement in the
                     * following form:
                     *
                     * EDGE_SE3:QUAT id1, id2, dx, dy, dz, dqx, dqy, dqz, dqw
                     *
                     * I11 I12 I13 I14 I15 I16
                     *     I22 I23 I24 I25 I26
                     *         I33 I34 I35 I36
                     *             I44 I45 I46
                     *                 I55 I56
                     *                     I66
                     */

                    // Extract formatted output
                    strstrm >> i >> j >> dx >> dy >> dz >> dqx >> dqy >> dqz >> dqw >> I11 >>
                            I12 >> I13 >> I14 >> I15 >> I16 >> I22 >> I23 >> I24 >> I25 >> I26 >>
                            I33 >> I34 >> I35 >> I36 >> I44 >> I45 >> I46 >> I55 >> I56 >> I66;

                    // Fill in elements of the measurement

                    // Pose ids
                    if (poses.insert({i, num_poses}).second) num_poses++;

                    if (poses.insert({j, num_poses}).second) num_poses++;
                    posemeasurement.i = poses[i];
                    posemeasurement.j = poses[j];
                    pose_ids.insert(i);
                    pose_ids.insert(j);

                    // Raw measurements
                    posemeasurement.t = Eigen::Matrix<Scalar, 3, 1>(dx, dy, dz);
                    posemeasurement.R =
                            Eigen::Quaternion<Scalar>(dqw, dqx, dqy, dqz).toRotationMatrix();

                    // Compute precisions

                    // Compute and store the optimal (information-divergence-minimizing) value
                    // of the parameter tau
                    Eigen::Matrix<Scalar, 3, 3> TranInfo;
                    TranInfo << I11, I12, I13, I12, I22, I23, I13, I23, I33;
                    posemeasurement.tau = 3 / TranInfo.inverse().trace();
                    posemeasurement.trans_precision = 3 / TranInfo.inverse().trace();
//                    measurement.trans_precision_true << TranInfo.inverse()(0, 0), TranInfo.inverse()(1, 1), TranInfo.inverse()(2, 2);
                    // Compute and store the optimal (information-divergence-minimizing value
                    // of the parameter kappa

                    Eigen::Matrix<Scalar, 3, 3> RotInfo;
                    RotInfo << I44, I45, I46, I45, I55, I56, I46, I56, I66;
                    posemeasurement.kappa = 3 / (2 * RotInfo.inverse().trace());
//                    measurement.kappa = 3 / (RotInfo.inverse().trace());
                    posemeasurement.rot_precision = 3 / ( RotInfo.inverse().trace());
//                    measurement.rot_precision_true << RotInfo.inverse()(0, 0), RotInfo.inverse()(1, 1), RotInfo.inverse()(2, 2);



                    measurements.poseMeasurements.push_back(posemeasurement);

                } else if (token == "LANDMARK2")
                {
                    strstrm >> i >> j >> dx >> dy >> I11 >> I12 >> I22;

                    if (poses.insert({i, num_poses}).second) num_poses++;

                    if (landmarks.insert({j, num_landmarks}).second) num_landmarks++;

                    landmarkmeasurement.i = poses[i];     // pose index (if you need a mapping there too, do the same)
                    landmarkmeasurement.j = landmarks[j];
                    pose_ids.insert(i);
                    landmark_ids.insert(j);    // the “j” is a landmark

                    landmarkmeasurement.l = Eigen::Matrix<Scalar,2,1>(dx,dy);

                    Eigen::Matrix<Scalar,2,2> TranCov;
                    TranCov << I11, I12,
                               I12, I22;
                    landmarkmeasurement.nu = 2.0 / TranCov.inverse().trace();

                    measurements.landmarkMeasurements.push_back(landmarkmeasurement);
                }else if (token == "LANDMARK3")
                {
                    strstrm >> i >> j >> dx >> dy >> dz >> dqx >> dqy >> dqz >> dqw >> I11 >>
                    I12 >> I13 >> I14 >> I15 >> I16 >> I22 >> I23 >> I24 >> I25 >> I26 >>
                    I33 >> I34 >> I35 >> I36 >> I44 >> I45 >> I46 >> I55 >> I56 >> I66;

                    if (poses.insert({i, num_poses}).second) num_poses++;

                    if (landmarks.insert({j, num_landmarks}).second) num_landmarks++;
                    landmarkmeasurement.i = poses[i];
                    landmarkmeasurement.j = landmarks[j];
                    pose_ids.insert(i);        // the “i” is still a pose
                    landmark_ids.insert(j);    // the “j” is a landmark
                    landmarkmeasurement.l = Eigen::Matrix<Scalar, 3, 1>(dx, dy, dz);

                    Eigen::Matrix<Scalar, 3, 3> TranCov;
                    TranCov << I11, I12, I13, I12, I22, I23, I13, I23, I33;

                    landmarkmeasurement.nu = 3 / (2 * TranCov.inverse().trace());



                    measurements.landmarkMeasurements.push_back(landmarkmeasurement);
                }else if ((token == "VERTEX_SE2") || (token == "VERTEX_SE3:QUAT")) {
                    // This is just initialization information, so do nothing
                    continue;
                }
                else {
                    std::cout << "Error: unrecognized type: " << token << "!" << std::endl;
                    assert(false);
                }


            } // while

            infile.close();
            num_poses_ = measurements.num_poses;
            return measurements;
        }

         Measurement read_pycfg_file(const std::string &filename) {
          std::ifstream infile(filename);
          if (!infile) {
              throw std::runtime_error("Could not open " + filename);
          }

          Measurement M;
          std::unordered_set<size_t> pose_ids, landmark_ids;

          std::string line;
          while (std::getline(infile, line)) {
                std::stringstream ss(line);
                std::string token;
                ss >> token;

                if (token == "VERTEX_SE2") {
                      // just record the pose index
                      std::string id; double x,y,theta;
                      ss >> id >> x >> y >> theta;
                      size_t idx = std::stoull(id.substr(1));
                      pose_ids.insert(idx);

                } else if (token == "VERTEX_XY") {
                      // just record the landmark index
                      std::string id; double x,y;
                      ss >> id >> x >> y;
                      size_t idx = std::stoull(id.substr(1));
                      landmark_ids.insert(idx);

                } else if (token == "EDGE_SE2") {
                      double timestamp;
                      std::string ida, idb;
                      double dx, dy, dtheta;
                      double I11,I12,I13,I22,I23,I33;
                      ss >> timestamp
                         >> ida >> idb
                         >> dx >> dy >> dtheta
                         >> I11 >> I12 >> I13 >> I22 >> I23 >> I33;

                      // decode indices
                      char ca = ida[0], cb = idb[0];
                      size_t ia = std::stoull(ida.substr(1)),
                             ib = std::stoull(idb.substr(1));
                      pose_ids.insert(ia);
                      pose_ids.insert(ib);

                      // fill measurement
                      RelativePoseMeasurement m;
                      m.ci = ca; m.cj = cb;
                      m.i  = ia; m.j  = ib;
                      // translation
                      m.t = Eigen::Matrix<Scalar,2,1>(dx, dy);
                      // rotation from angle
                      m.R = Eigen::Rotation2D<Scalar>(dtheta).toRotationMatrix();
                      // information → tau, kappa
                      Eigen::Matrix2d Tinfo;
                      Tinfo << I11, I12,
                               I12, I22;
                      m.tau   = 2.0 / Tinfo.trace();
                      m.kappa = 1.0 / I33;

                      M.poseMeasurements.push_back(m);

                } else if (token == "EDGE_RANGE") {
                      double timestamp;
                      std::string ida, idl;
                      double range, sigma;
                      ss >> timestamp
                         >> ida >> idl
                         >> range >> sigma;

                      char ca = ida[0], cl = idl[0];
                      size_t ia = std::stoull(ida.substr(1)),
                             il = std::stoull(idl.substr(1));
                      pose_ids.insert(ia);
                      landmark_ids.insert(il);

                      RangeMeasurement m;
                      m.ci    = ca;   m.cj    = cl;
                      m.i     = ia;   m.j     = il;
                      m.range = range;
                      m.sigma = sigma;

                      M.rangeMeasurements.push_back(m);
                }
          }
          infile.close();

          // zero‑based indexing → count = maxID + 1
          M.num_poses     = pose_ids.size();
          M.num_landmarks = landmark_ids.size();
          M.num_ranges    = M.rangeMeasurements.size();

          return M;
    }


    }// namespace DataParser
} // namespace gtsam