//
// Created by jason on 4/30/25.
//
#ifndef STIEFELMANIFOLDEXAMPLE_CERTIFIABLERA_H
#define STIEFELMANIFOLDEXAMPLE_CERTIFIABLERA_H
#pragma once
#include "Certifiable_problem.h"

/**
 * @brief Certifiable Rotation Averaging (RA) problem using Stiefel manifold relaxations.
 *
 * Implements a hierarchy of relaxations for rotation averaging in SO(d),
 * certifies solutions using eigenvalue tests, and refines by lifting when needed.
 *
 * @tparam d Ambient dimension (must be 2 or 3).
 */
template <size_t d>
class CertifiableRA : public CertifiableProblem {
    static_assert(d == 2 || d == 3, "CertifiableRA only supports d = 2 or 3.");

public:
    /**
     * @brief Constructor.
     * @param p             Initial relaxation rank.
     * @param measurements  Parsed measurements (relative rotations).
     */
    CertifiableRA(size_t p, const DataParser::Measurement& measurements)
        : CertifiableProblem(d, p, measurements)
    {
        certificateResults_.startingRank = p;
    }

    /**
     * @brief Initialize problem: build graph, recover data matrix, random init, record time.
     */
    void init() {
        auto t0 = std::chrono::high_resolution_clock::now();
        currentGraph_     = buildGraphAtLevel(currentRank_);
        M_                = recoverDataMatrixFromGraph();
        currentValues_    = randomInitAtLevelP(currentRank_);
        auto t1 = std::chrono::high_resolution_clock::now();
        certificateResults_.initialization_time.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count()
        );
    }

    /**
     * @brief Build factor graph of relative rotation measurements at level p.
     * @param p Relaxation rank.
     * @return  NonlinearFactorGraph containing RaFactor2 or RaFactor3 factors.
     */
    NonlinearFactorGraph buildGraphAtLevel(size_t p) override {
        NonlinearFactorGraph inputGraph;
        for (const auto& meas : measurements_.poseMeasurements) {
            double Kappa = meas.kappa;
            Vector sigmas(p * d);
            sigmas.setConstant(std::sqrt(1.0 / (2 * Kappa)));
            auto noise = noiseModel::Diagonal::Sigmas(sigmas);

            if constexpr (d == 2) {
                inputGraph.emplace_shared<RaFactor2>(
                    meas.i, meas.j,
                    Rot2::atan2(meas.R(1, 0), meas.R(0, 0)),
                    p, noise
                );
            } else {
                inputGraph.emplace_shared<RaFactor3>(
                    meas.i, meas.j,
                    Rot3(meas.R),
                    p, noise
                );
            }
        }
        return inputGraph;
    }

    /**
     * @brief Compute dense block‐diagonal Λ blocks from Y.
     * @param Y Variable matrix.
     * @return  Matrix Lambda Blocks
     */
    Matrix computeLambdaBlocks(const Matrix& Y) override {
        Matrix SY = M_ * Y;
        Matrix Yt = Y.transpose();
        Matrix LambdaBlocks(d, num_pose_ * d);

        for (size_t i = 0; i < num_pose_; ++i) {
            Matrix P = SY.block(i * d, 0, d, Y.cols())
                     * Yt.block(0, i * d, Y.cols(), d);
            LambdaBlocks.block(0, i * d, d, d) = 0.5 * (P + P.transpose());
        }
        return LambdaBlocks;
    }

    /**
     * @brief Convert dense Λ blocks into a sparse Λ matrix.
     * @param LambdaBlocks  Dense block matrix.
     * @return               Sparse certificate matrix Λ.
     */
    SparseMatrix computeLambdaFromLambdaBlocks(const Matrix& LambdaBlocks) override {
        std::vector<Eigen::Triplet<Scalar>> elements;
        elements.reserve(d * d * num_pose_);
        for (size_t i = 0; i < num_pose_; ++i) {
            for (size_t r = 0; r < d; ++r) {
                for (size_t c = 0; c < d; ++c) {
                    elements.emplace_back(
                        i * d + r,
                        i * d + c,
                        LambdaBlocks(r, i * d + c)
                    );
                }
            }
        }
        SparseMatrix Lambda(d * num_pose_, d * num_pose_);
        Lambda.setFromTriplets(elements.begin(), elements.end());
        return Lambda;
    }

    /**
     * @brief Assemble sparse element matrix S from current Values.
     * @param values GTSAM Values of StiefelManifoldKP variables.
     * @return       SparseMatrix S.
     */
    SparseMatrix elementMatrix(const Values& values) override {
        const size_t N = values.size();
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(N * currentRank_ * d);

        for (const auto& kv : values.extract<StiefelManifoldKP>()) {
            size_t i = kv.first;
            const auto& mat = kv.second.matrix();  // p × d
            for (size_t row = 0; row < currentRank_; ++row) {
                for (size_t col = 0; col < d; ++col) {
                    triplets.emplace_back(i * d + col, row, mat(row, col));
                }
            }
        }

        SparseMatrix S(N * d, currentRank_);
        S.setFromTriplets(triplets.begin(), triplets.end());
        return S;
    }

    /**
     * @brief Randomly initialize Stiefel variables at level Pmin.
     * @param Pmin Relaxation rank.
     * @return     Values container with random StiefelManifoldKP entries.
     */
    Values randomInitAtLevelP(const size_t Pmin) override {
        Values initial;
        for (size_t j = 0; j < num_pose_; ++j) {
            StiefelManifoldKP Y = StiefelManifoldKP::Random(
                std::default_random_engine::default_seed, d, Pmin
            );
            initial.insert(j, Y);
        }
        return initial;
    }

    /**
     * @brief Convert flat vector v into VectorValues on tangent spaces.
     * @param p      Relaxation rank.
     * @param v      Flat descent vector.
     * @param values Basepoint Values at level p.
     * @return       VectorValues mapping each variable to its tangent vector.
     */
    VectorValues TangentVectorValues(size_t p, const Vector v, const Values values) override {
        VectorValues delta;
        Matrix Ydot = Matrix::Zero(v.size(), p);
        Ydot.rightCols<1>() = v;

        for (const auto& kv : values.extract<StiefelManifoldKP>()) {
            size_t idx = gtsam::symbolIndex(kv.first);
            const auto& pose = kv.second;
            Matrix tangM = Ydot.block(idx * d, 0, d, p).transpose();
            Vector xi   = StiefelManifoldKP::Vectorize(tangM);
            Vector tang = pose.G_.transpose() * xi;
            delta.insert(idx, tang);
        }
        return delta;
    }

    /**
     * @brief Project ambient gradient Ydot onto tangent space at Y.
     * @param p    Relaxation rank.
     * @param Y    Basepoint matrix.
     * @param Ydot Ambient gradient.
     * @return     Tangent‐space projection.
     */
    Matrix tangent_space_projection(const size_t p,
                                    const Matrix& Y,
                                    const Matrix& Ydot) override {
        Matrix result = Ydot;
        size_t total = d * num_pose_;
        result.block(0, 0, total, p) =
            StiefelManifoldKP::Proj(
                Y.block(0, 0, total, p).transpose(),
                result.block(0, 0, total, p).transpose()
            ).transpose();
        return result;
    }

    /**
     * @brief Recover the data matrix from the factor graph.
     * @return SparseMatrix L of size.
     */
    SparseMatrix recoverDataMatrixFromGraph() override {
        std::vector<Eigen::Triplet<Scalar>> triplets;
        triplets.reserve(currentGraph_.size() * (4 * d + 2 * d * d));

        if (d == 2) {
            for (const auto& f_ptr : currentGraph_) {
                if (auto ra = std::dynamic_pointer_cast<RaFactor2>(f_ptr)) {
                    const auto& blk = ra->getRotationConnLaplacianBlock();
                    triplets.insert(triplets.end(), blk.begin(), blk.end());
                }
            }
        } else {
            for (const auto& f_ptr : currentGraph_) {
                if (auto ra = std::dynamic_pointer_cast<RaFactor3>(f_ptr)) {
                    const auto& blk = ra->getRotationConnLaplacianBlock();
                    triplets.insert(triplets.end(), blk.begin(), blk.end());
                }
            }
        }

        SparseMatrix L(d * num_pose_, d * num_pose_);
        L.setFromTriplets(triplets.begin(), triplets.end(), std::plus<Scalar>());
        L.makeCompressed();
        return L;
    }

    /**
     * @brief Solve the rotation averaging problem for p ∈ [pMin, pMax].
     * @param pMin Minimum rank.
     * @param pMax Maximum rank.
     * @return     CertificateResults on success, std::nullopt otherwise.
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
     * @brief Round the solution back to problem dimension.
     * @return matrix R.
     */
    Matrix RoundSolutionS() override {
        Matrix S = elementMatrix(currentValues_).transpose();
        Eigen::JacobiSVD<Matrix> svd(S, Eigen::ComputeFullV);
        Vector sigmas = svd.singularValues();

        // Build truncated Sigma
        DiagonalMatrix Sigma_d(d);
        for (size_t i = 0; i < d; ++i)
            Sigma_d.diagonal()(i) = sigmas(i);

        Matrix R = Sigma_d * svd.matrixV().leftCols(d).transpose();
        size_t n_ = num_pose_;
        Vector dets(n_);
        size_t rot_offset = 0;
        size_t ng0 = 0;

        for (size_t i = 0; i < n_; ++i) {
            dets(i) = R.block(0, rot_offset + i * d, d, d).determinant();
            if (dets(i) > 0) ++ng0;
        }
        if (ng0 < n_ / 2) {
            Matrix reflector = Matrix::Identity(d, d);
            reflector(d - 1, d - 1) = -1;
            R = reflector * R;
        }
        for (size_t i = 0; i < n_; ++i) {
            R.block(0, rot_offset + i * d, d, d) =
                project_to_SOd(R.block(0, rot_offset + i * d, d, d));
        }
        return R;
    }

    /**
     * @brief Export solution in G2O or TUM format.
     * @param path File base name.
     * @param R    Rotation/translation matrix.
     * @param g2o  True → G2O, False → TUM text.
     */
    void ExportData(const string& path, Matrix& R, bool g2o) override {
        if (g2o) {
            Values finalposes;
            if (d == 3) {
                for (size_t i = 0; i < num_pose_; ++i) {
                    finalposes.insert(i, Rot3(R.block(0,i * 3, 3, 3)));
                }
            } else {
                for (size_t i = 0; i < num_pose_; ++i) {
                    Rot2 rot = Rot2::fromCosSin(
                        R.block(0,  i * 2, 2, 2)(0,0),
                        R.block(0, i * 2, 2, 2)(1,0)
                    );
                    finalposes.insert(i, rot);
                }
            }
            writeG2o(currentGraph_, finalposes, path + ".g2o");
        } else {
            std::ofstream file(path + ".txt");
            if (d == 2) {
                for (size_t i = 0; i < num_pose_; ++i) {
                    Eigen::Matrix3d R3 = Eigen::Matrix3d::Identity();
                    R3.topLeftCorner<2,2>() = R.block(0, i * 2, 2, 2);
                    auto q = Eigen::Quaterniond(R3);
                    file << i << " " << q.x() << " " << q.y()
                         << " " << q.z() << " " << q.w() << "\n";
                }
            } else {
                for (size_t i = 0; i < num_pose_; ++i) {
                    Quaternion q(R.block<3,3>(0, i * 3));
                    file << i << " " << q.x() << " " << q.y()
                         << " " << q.z() << " " << q.w() << "\n";
                }
            }
            file.close();
        }
    }
};


// --- Type aliases for easy use
using CertifiableRA2 = CertifiableRA<2>;
using CertifiableRA3 = CertifiableRA<3>;

#endif // STIEFELMANIFOLDEXAMPLE_CERTIFIABLERA_H
