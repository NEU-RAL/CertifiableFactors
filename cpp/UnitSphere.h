/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */
#ifndef STIEFELMANIFOLDEXAMPLE_UNITSPHERE_H
#define STIEFELMANIFOLDEXAMPLE_UNITSPHERE_H
#pragma once
#include "StiefelManifold.h"
#include "StiefelManifold-inl.h"
/**
 * @file UnitSphere.h
 * @date
 * @author Jason Xu
 * @brief
 */
namespace gtsam {

    /**
     * @brief Alias for the unit sphere in ℝᴷ as a Stiefel manifold.
     *
     * The unit sphere S⁽ᴷ⁻¹⁾ can be represented as the Stiefel manifold
     * St(K, 1), i.e., the set of K×1 orthonormal frames.
     *
     * @tparam K  Ambient dimension of the unit sphere.
     */
    template <int P>
    using UnitSphere = StiefelManifold<1, P>;

    /**
     * @brief Dynamic‐size alias for the unit sphere manifold.
     *
     * Both the ambient dimension (rows) and the manifold dimension (columns)
     * can be specified at runtime via Eigen::Dynamic.
     */
//    using UnitSphereD = StiefelManifold<Eigen::Dynamic, Eigen::Dynamic>;
     using UnitSphereD = UnitSphere<Eigen::Dynamic>;


}  // namespace gtsam







#endif //STIEFELMANIFOLDEXAMPLE_UNITSPHERE_H


