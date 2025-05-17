//
// Created by jason on 4/30/25.
//

#ifndef STIEFELMANIFOLDEXAMPLE_CERTIFIABLEPGO_H
#define STIEFELMANIFOLDEXAMPLE_CERTIFIABLEPGO_H
#pragma once
#include "Certifiable_problem.h"

/**
 * @brief Certifiable Pose Graph Optimization (PGO)
 *
 * Implements a hierarchy of relaxations of PGO via rank‐d Stiefel manifolds,
 * certifies solutions using eigenvalue tests, and refines by lifting along
 * descent directions when necessary.
 *
 * @tparam d  Ambient pose dimension (must be 2 or 3).
 */
template <size_t d>
class CertifiablePGO : public CertifiableProblem {
    static_assert(d == 2 || d == 3, "CertifiablePGO only supports d = 2 or 3.");

public:
    /**
     * @brief Construct with initial rank and measurement data.
     * @param p             Initial relaxation rank.
     * @param measurements  Parsed measurement struct (num_poses, etc.).
     */
    CertifiablePGO(size_t p, const DataParser::Measurement& measurements)
        : CertifiableProblem(d, p, measurements)
    {
        certificateResults_.startingRank = p;
    }

    /**
     * @brief Initialize graph, data matrix, and random values; record init time.
     */
    void init() {
        auto t0 = std::chrono::high_resolution_clock::now();
        currentGraph_   = buildGraphAtLevel(currentRank_);
        M_              = recoverDataMatrixFromGraph();
        currentValues_  = randomInitAtLevelP(currentRank_);
        auto t1 = std::chrono::high_resolution_clock::now();
        certificateResults_.initialization_time.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count()
        );
    }

    /**
     * @brief Assemble the factor graph at relaxation level p.
     * @param p  Relaxation rank.
     * @return   Factor graph containing only pose‐to‐pose factors.
     */
    NonlinearFactorGraph buildGraphAtLevel(size_t p) override {
        NonlinearFactorGraph inputGraph;
        for (const auto& meas : measurements_.poseMeasurements) {
            // Construct diagonal noise sigmas from kappa and tau
            Vector sigmas = Vector::Zero(p * d + p);
            sigmas.head(p * d).setConstant(std::sqrt(1.0 / (2 * meas.kappa)));
            sigmas.tail(p).setConstant(std::sqrt(1.0 / (2 * meas.tau)));
            auto noise = noiseModel::Diagonal::Sigmas(sigmas);

            // Emplace the appropriate SE‐sync factor
            if constexpr (d == 2) {
                inputGraph.emplace_shared<SEsyncFactor2>(
                    meas.i, meas.j, meas.R, meas.t, p, noise
                );
            } else {
                inputGraph.emplace_shared<SEsyncFactor3>(
                    meas.i, meas.j, meas.R, meas.t, p, noise
                );
            }
        }
        return inputGraph;
    }

    /**
     * @brief Compute the dense block‐diagonal certificate matrix.
     * @param Y  Variable matrix.
     * @return   Dense block matrix of size (d × d·num_poses).
     */
    Matrix computeLambdaBlocks(const Matrix& Y) override {
        Matrix SY = M_ * Y;
        Matrix Yt = Y.transpose();
        Matrix LambdaBlocks(d, num_pose_ * d);
        size_t offset = num_pose_;

        for (size_t i = 0; i < num_pose_; ++i) {
            Matrix P = SY.block(offset + i * d, 0, d, Y.cols())
                     * Yt.block(0, offset + i * d, Y.cols(), d);
            LambdaBlocks.block(0, i * d, d, d) = 0.5 * (P + P.transpose());
        }
        return LambdaBlocks;
    }

    /**
     * @brief Convert dense Λ_blocks into a sparse Λ matrix for certification.
     * @param LambdaBlocks  Dense block matrix.
     * @return              Sparse certificate matrix Λ.
     */
    SparseMatrix computeLambdaFromLambdaBlocks(
        const Matrix& LambdaBlocks) override
    {
        std::vector<Eigen::Triplet<Scalar>> elements;
        elements.reserve(d * d * num_pose_);
        size_t offset = num_pose_;
        for (size_t i = 0; i < num_pose_; ++i) {
            for (size_t r = 0; r < d; ++r) {
                for (size_t c = 0; c < d; ++c) {
                    elements.emplace_back(
                        offset + i * d + r,
                        offset + i * d + c,
                        LambdaBlocks(r, i * d + c)
                    );
                }
            }
        }
        SparseMatrix Lambda(offset + d * num_pose_, offset + d * num_pose_);
        Lambda.setFromTriplets(elements.begin(), elements.end());
        return Lambda;
    }

    /**
     * @brief Build the element matrix S from current Values.
     * @param values  GTSAM Values containing LiftedPoseDP variables.
     * @return        Sparse matrix S of size.
     */
    SparseMatrix elementMatrix(const Values& values) override {
        const size_t N = num_pose_, p = currentRank_;
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(N * p * (d + 1));

        for (const auto& kv : values.extract<LiftedPoseDP>()) {
            size_t i = kv.first;
            const auto& pose = kv.second;
            const auto& t   = pose.get_TranslationVector(); // p × 1
            const auto& mat = pose.matrix();                // p × d

            // Translation entries
            for (size_t row = 0; row < p; ++row)
                triplets.emplace_back(i, row, t(row));

            // Rotation (Stiefel) entries
            size_t row0 = N + i * d;
            for (size_t row = 0; row < p; ++row)
                for (size_t col = 0; col < d; ++col)
                    triplets.emplace_back(row0 + col, row, mat(row, col));
        }

        SparseMatrix S(N + N * d, p);
        S.setFromTriplets(triplets.begin(), triplets.end());
        return S;
    }

    /**
     * @brief Randomly initialize all poses at level Pmin using uniform Stiefel and random translation.
     * @param Pmin  Target relaxation rank.
     * @return      GTSAM Values container with random LiftedPoseDP entries.
     */
    Values randomInitAtLevelP(const size_t Pmin) override {
        Values initial;
        for (size_t j = 0; j < num_pose_; ++j) {
            StiefelManifoldKP Y =
                StiefelManifoldKP::Random(std::default_random_engine::default_seed, d, Pmin);
            Vector trans = Vector::Random(Pmin);
            initial.insert(j, LiftedPoseDP(Y, trans));
        }
        return initial;
    }

    /**
     * @brief Convert a flat eigenvector into tangent VectorValues.
     * @param p  Relaxation rank.
     * @param v  Flattened direction of size.
     * @param values  Lifted Values at level p.
     * @return  VectorValues on each local tangent space.
     */
    VectorValues TangentVectorValues(
        size_t p, const Vector v, const Values values) override
    {
        VectorValues delta;
        Matrix Ydot = Matrix::Zero(v.size(), p);
        Ydot.rightCols<1>() = v;

        for (const auto& kv : values.extract<LiftedPoseDP>()) {
            size_t idx = gtsam::symbolIndex(kv.first);
            const auto& Y = kv.second.get_Y();

            // Extract ambient gradient blocks
            Matrix tangM = Ydot.block(num_pose_ + idx * d, 0, d, p).transpose();
            Vector transV = Ydot.block(idx, 0, 1, p).transpose();

            Vector xi = StiefelManifoldKP::Vectorize(tangM);
            Vector tVec = Y.G_.transpose() * xi;

            Vector combined(tVec.size() + transV.size());
            combined << tVec, transV;
            delta.insert(idx, combined);
        }
        return delta;
    }

    /**
     * @brief Project ambient‐space variation Ydot onto the tangent space at Y.
     * @param p     Relaxation rank.
     * @param Y     Basepoint matrix.
     * @param Ydot  Ambient variation.
     * @return      Tangent‐space projection.
     */
    Matrix tangent_space_projection(
        const size_t p, const Matrix& Y, const Matrix& Ydot) override
    {
        Matrix result = Ydot;
        size_t offset_t  = num_pose_;
        size_t rot_mat_sz = d * num_pose_;
        result.block(offset_t, 0, rot_mat_sz, p) =
            StiefelManifoldKP::Proj(
                Y.block(0, 0, rot_mat_sz, p).transpose(),
                result.block(0, 0, rot_mat_sz, p).transpose()
            ).transpose();
        return result;
    }

    /**
     * @brief Recover the data matrix from the current factor graph.
     * @return Sparse data matrix L of size.
     */
    SparseMatrix recoverDataMatrixFromGraph() override {
        std::vector<Eigen::Triplet<Scalar>> triplets;
        triplets.reserve(currentGraph_.size() * (4 + 2*d*d + 2*d + 2));

        if (d == 2) {
            for (auto& f_ptr : currentGraph_) {
                if (auto factor = std::dynamic_pointer_cast<SEsyncFactor2>(f_ptr))
                    factor->appendBlocksFromFactor(num_pose_, triplets);
            }
        } else {
            for (auto& f_ptr : currentGraph_) {
                if (auto factor = std::dynamic_pointer_cast<SEsyncFactor3>(f_ptr))
                    factor->appendBlocksFromFactor(num_pose_, triplets);
            }
        }

        SparseMatrix L((d + 1) * num_pose_, (d + 1) * num_pose_);
        L.setFromTriplets(triplets.begin(), triplets.end());
        return L;
    }

    /**
     * @brief Search for a certifiable solution between pMin and pMax.
     *
     * Performs LM optimization, gradient checks, and fast_verification at each level.
     * Lifts along descent if verification fails. Returns certificate results on success.
     *
     * @param pMin  Minimum relaxation rank.
     * @param pMax  Maximum relaxation rank.
     * @return      CertificateResults on success, or std::nullopt if none found.
     */
    std::optional<CertificateResults> Solve(size_t pMin, size_t pMax) {
        Values Qstar;
        auto t6 = std::chrono::high_resolution_clock::now();
        for (size_t p = pMin; p <= pMax; ++p) {
            std::cout << "Starting optimization at rank = " << p << std::endl;
            auto t0 = std::chrono::high_resolution_clock::now();
            setCurrentRank(p);
            Qstar = tryOptimizingAtLevel(p);
            setCurrentValues(Qstar);
            auto t1 = std::chrono::high_resolution_clock::now();
            certificateResults_.elapsed_optimization_times.push_back((std::chrono::duration<double, std::milli> (t1 - t0)).count());

            auto t2 = std::chrono::high_resolution_clock::now();
            auto nonlinear_graph = buildGraphAtLevel(p);

            auto linear_graph = nonlinear_graph.linearize(Qstar);
            auto grad_norm = linear_graph->gradientAtZero();
            std::cout << "Gradient norm at level p = " << p << " is : " << grad_norm.norm() << std::endl;
            certificateResults_.gradnorm.push_back(grad_norm.norm());
            auto t3 = std::chrono::high_resolution_clock::now();
            certificateResults_.initialization_time.push_back((std::chrono::duration<double, std::milli> (t3 - t2)).count());

            auto t4 = std::chrono::high_resolution_clock::now();
            SparseMatrix S = elementMatrix(Qstar);
            Matrix lambdaBlocks = computeLambdaBlocks(S);
            SparseMatrix Lambda = computeLambdaFromLambdaBlocks(lambdaBlocks);
            SparseMatrix M = getDataMatrix();

            bool success = false;
            Scalar eta = opts_.eta;
            size_t nx = opts_.nx;
            Vector v;
            Scalar theta;
            size_t num_lobpcg_iters;
            size_t max_iters = opts_.max_iters;
            Scalar max_fill_factor =  opts_.max_fill_factor;
            Scalar drop_tol = opts_.drop_tol;

            success = fast_verification(M - Lambda, eta, nx, theta, v,
                                        num_lobpcg_iters, max_iters, max_fill_factor, drop_tol);
            auto t5 = std::chrono::high_resolution_clock::now();
            certificateResults_.verification_times.push_back((std::chrono::duration<double, std::milli> (t5 - t4)).count());
            if (!success) {
                increaseCurrentRank();
                currentValues_ = initializeWithDescentDirection(Qstar, M, v, theta, 1e-2);
            } else {
                std::cout << "Solution verified at level p = " << p << std::endl;
                certificateResults_.Yopt = S;
                certificateResults_.Lambda = Lambda;
                certificateResults_.xhat = RoundSolutionS();
                auto t7 = std::chrono::high_resolution_clock::now();
                certificateResults_.total_computation_time = (std::chrono::duration<double, std::milli> (t7 - t6).count());
                certificateResults_.endingRank = p;
                return certificateResults_;
            }
        }

        std::cout << "No certifiable solution found in p ∈ [" << pMin << ", " << pMax << "]" << std::endl;
        return std::nullopt;
    }

    /**
     * @brief Round the relaxed solution back to problem dimension.
     * @return Matrix R.
     */
    Matrix RoundSolutionS() override {
        Matrix S = elementMatrix(currentValues_).transpose();


        // First, compute a thin SVD of Y
        Eigen::JacobiSVD<Matrix> svd(S, Eigen::ComputeFullV);

        Vector sigmas = svd.singularValues();

        // Construct a diagonal matrix comprised of the first d singular values
        DiagonalMatrix Sigma_d(d);
        DiagonalMatrix::DiagonalVectorType &diagonal = Sigma_d.diagonal();
        for (size_t i = 0; i < d; ++i)
            diagonal(i) = sigmas(i);

        // First, construct a rank-d truncated singular value decomposition for Y
        Matrix R = Sigma_d * svd.matrixV().leftCols(d).transpose();
        Vector determinants(num_pose_);

        // Compute the offset at which the rotation matrix blocks begin
        size_t rot_offset;
        rot_offset = num_pose_;


        size_t ng0 = 0; // This will count the number of blocks whose
        // determinants have positive sign
        for (size_t i = 0; i < num_pose_; i++) {
            // Compute the determinant of the ith dxd block of R
            determinants(i) = R.block(0, rot_offset + i * d, d, d).determinant();
            if (determinants(i) > 0)
                ++ng0;
        }

        if (ng0 < num_pose_ / 2) {
            // Less than half of the total number of blocks have the correct sign, so
            // reverse their orientations

            // Get a reflection matrix that we can use to reverse the signs of those
            // blocks of R that have the wrong determinant
            Matrix reflector = Matrix::Identity(d, d);
            reflector(d - 1, d - 1) = -1;

            R = reflector * R;
        }

        // Finally, project each dxd rotation block to SO(d)
        for (size_t i = 0; i < num_pose_; i++) {
            R.block(0, rot_offset + i * d, d, d) = project_to_SOd(R.block(0, rot_offset + i * d, d, d));
        }
        return R;
    }

    /**
     * @brief Export the solution in G2O or TUM format.
     * @param path  Base filename (without extension).
     * @param R     Rotation/translation solution matrix.
     * @param g2o   If true, write G2O; otherwise, write TUM.
     */
    void ExportData(const string &path, Matrix &R, bool g2o) override {
        if (g2o) {
            Values finalposes;
            if (d == 3) {
                // Insert SE3 poses
                for (auto i = 0; i < num_pose_; ++i) {
                    finalposes.insert(
                        i,
                        Pose3(
                            Rot3(R.block(0, num_pose_ + i * d, d, d)),
                            R.block(0, i, d, 1)
                        )
                    );
                }
            } else {
                // Insert SE2 poses
                for (auto i = 0; i < num_pose_; ++i) {
                    Rot2 rot = Rot2::fromCosSin(
                        R.block(0, num_pose_ + i * d, d, d)(0, 0),
                        R.block(0, num_pose_ + i * d, d, d)(1, 0)
                    );
                    finalposes.insert(i, Pose2(rot, R.block(0, i, d, 1)));
                }
            }
            writeG2o(currentGraph_, finalposes, path + ".g2o");
        } else {
            std::ofstream file(path + ".txt");
            if (d == 2) {
                for (auto i = 0; i < num_pose_; ++i) {
                    Eigen::Matrix3d R3 = Eigen::Matrix3d::Identity();
                    R3.topLeftCorner<2,2>() =
                        R.block(0, num_pose_ + i*2, 2, 2);
                    auto q = Eigen::Quaterniond(R3);
                    Vector t = R.block(0, i, d, 1);
                    file << i << " " << t(0) << " " << t(1)
                         << " " << q.x() << " " << q.y()
                         << " " << q.z() << " " << q.w() << "\n";
                }
            } else {
                for (auto i = 0; i < num_pose_; ++i) {
                    Quaternion q(R.block<3,3>(0, num_pose_ + i*3));
                    Vector t = R.block(0, i, d, 1);
                    file << i << " " << t(0) << " " << t(1)
                         << " " << t(2) << " " << q.x() << " "
                         << q.y() << " " << q.z() << " "
                         << q.w() << "\n";
                }
            }
            file.close();
        }
    }
};


// Convenience aliases
using CertifiablePGO2 = CertifiablePGO<2>;
using CertifiablePGO3 = CertifiablePGO<3>;

#endif // STIEFELMANIFOLDEXAMPLE_CERTIFIABLEPGO_H

