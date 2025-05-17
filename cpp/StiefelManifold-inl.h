/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/*
 * @file StiefelManifold.h
 * @date Jan 1 ,2025
 * @author Zhexin Xu, Nikolas Sanderson
 * @brief  Template implementations of StiefelManifold
 */
#ifndef STIEFELMANIFOLD_STIEFELMANIFOLD_INL_H
#define STIEFELMANIFOLD_STIEFELMANIFOLD_INL_H
#pragma once
#include "StiefelManifold.h"
#include <gtsam/base/Matrix.h>

namespace gtsam {

template<int K_, int P_>
StiefelManifold<K_, P_> StiefelManifold<K_, P_>::projectToManifold(const gtsam::Matrix &A)
{
    size_t p = A.rows();
    size_t k = A.cols();
    gtsam::Matrix P(p,k);

    if (A.rows() != p || A.cols() != k)
    {
        throw std::runtime_error("Error in projectToManifold");
    }

    Eigen::JacobiSVD<gtsam::Matrix> SVD(A.block(0,0,p,k),Eigen::ComputeThinU | Eigen::ComputeThinV);
    P.block(0,0, p,k) = SVD.matrixU() * SVD.matrixV().transpose();

    return StiefelManifold<K_, P_>::FromMatrix(P);
}

template<int K_, int P_>
Matrix StiefelManifold<K_, P_>::SymBlockDiagProduct(const gtsam::Matrix &A, const gtsam::Matrix &BT, const gtsam::Matrix &C)
{
    size_t p = A.rows(), k = A.cols();
    gtsam::Matrix R(p, k);
    gtsam::Matrix P(k,k);
    gtsam::Matrix S(k,k);

    P = BT.block(0,0,k,p) * C.block(0,0,p,k);
    S = 0.5 * (P + P.transpose());
    R.block(0,0,p,k) =  A.block(0,0,p,k) * S;

    return R;
}
template<int K_, int P_>
StiefelManifold<K_, P_> StiefelManifold<K_, P_>::random_sample(const std::default_random_engine::result_type &seed) const
{
    std::default_random_engine generator(seed);
    std::normal_distribution<double> g;

    Matrix R(p_,k_);
    for (size_t r = 0; r<p_;++r)
        for (size_t c = 0; c< k_; ++c)
            R(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = g(generator);

    return projectToManifold(R);
}

    template<int K_, int P_>
    StiefelManifold<K_, P_> StiefelManifold<K_, P_>::Random(const std::default_random_engine::result_type &seed, size_t k, size_t p)
    {
        std::default_random_engine generator(seed);
        std::normal_distribution<double> g;

        Matrix R(p,k);
        for (size_t r = 0; r<p;++r)
            for (size_t c = 0; c< k; ++c)
                R(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = g(generator);

        return projectToManifold(R);
    }

template<int K_, int P_>
void StiefelManifold<K_, P_>::print(const std::string &s) const {
    std::cout << s << matrix_ << std::endl;
}


} // namespace gtsam





#endif //STIEFELMANIFOLD_STIEFELMANIFOLD_INL_H
