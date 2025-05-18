/* ----------------------------------------------------------------------------
 * Copyright 2025, Northeastern University Robust Autonomy Lab, * Boston, MA 02139
 * All Rights Reserved
 * Authors: Zhexin Xu, Nikolas Sanderson
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef STIEFELMANIFOLDEXAMPLE_RELATIVEPOSEMEASUREMENT_H
#define STIEFELMANIFOLDEXAMPLE_RELATIVEPOSEMEASUREMENT_H

#include <Eigen/Dense>
#include <iostream>

#include "types.h"

using namespace std;

namespace gtsam {
namespace DataParser {

/**
 * @brief A simple struct that contains the elements of a relative pose measurement.
 *
 * Stores a 2D/3D relative transformation between two poses along with
 * associated rotational and translational precision values.
 */
  struct RelativePoseMeasurement {
      /** @brief 0-based index of the first pose. */
      char ci;
      size_t i;

      /** @brief 0-based index of the second pose. */
      char cj;
      size_t j;

      /** @brief Rotational measurement matrix R. */
      Matrix R;

      /** @brief Translational measurement vector t. */
      Vector t;

      /** @brief Concentration (precision) for the rotational measurement. */
      Scalar kappa;

      /** @brief Precision for the translational measurement. */
      Scalar tau;

      /** @brief (Deprecated) Rotational measurement precision. */
      Scalar rot_precision;

      /** @brief (Deprecated) Translational measurement precision. */
      Scalar trans_precision;

      /** @brief True rotational precision vector (optional). */
      Vector rot_precision_true;

      /** @brief True translational precision vector (optional). */
      Vector trans_precision_true;

      /** @brief Default constructor. Leaves fields uninitialized. */
      RelativePoseMeasurement() {}

      /**
       * @brief Constructs a RelativePoseMeasurement with given data.
       *
       * @param first_pose             Index of the first pose.
       * @param second_pose            Index of the second pose.
       * @param relative_rotation      Rotation matrix between poses.
       * @param relative_translation   Translation vector between poses.
       * @param rotational_precision   Rotational precision (kappa).
       * @param translational_precision Translational precision (tau).
       */
      RelativePoseMeasurement(size_t first_pose, size_t second_pose,
                              const Matrix& relative_rotation,
                              const Vector& relative_translation,
                              Scalar rotational_precision,
                              Scalar translational_precision)
          : i(first_pose),
            j(second_pose),
            R(relative_rotation),
            t(relative_translation),
            kappa(rotational_precision),
            tau(translational_precision) {}

      /**
       * @brief Stream operator for easy printing of the measurement.
       *
       * @param os           Output stream.
       * @param measurement  Measurement to print.
       * @return             Reference to the output stream.
       */
      inline friend std::ostream& operator<<(std::ostream& os,
                                             const RelativePoseMeasurement& measurement) {
        os << "i: " << measurement.i << std::endl;
        os << "j: " << measurement.j << std::endl;
        os << "R: " << std::endl << measurement.R << std::endl;
        os << "t: " << std::endl << measurement.t << std::endl;
        os << "Kappa: " << measurement.kappa << std::endl;
        os << "Tau: " << measurement.tau << std::endl;
        return os;
      }
  };

/** @brief Typedef for a vector of RelativePoseMeasurement structs. */
typedef std::vector<RelativePoseMeasurement> measurements_t;

/**
 * @brief A struct representing a simple range measurement between a pose and a landmark.
 *
 * Stores the range value and its associated noise standard deviation.
 */
  struct RangeMeasurement {
      /** @brief Index of the pose. */
      char ci;
      size_t i;

      /** @brief Index of the landmark. */
      char cj;
      size_t j;

      /** @brief Measured range between pose and landmark. */
      Scalar range;

      /** @brief Noise presicion of the range measurement. */
      Scalar sigma;

      /** @brief Default constructor. Leaves fields uninitialized. */
      RangeMeasurement() {}

      /**
       * @brief Constructs a RangeMeasurement with given data.
       *
       * @param pose_index  Index of the pose.
       * @param Lmk_index   Index of the landmark.
       * @param range_      Measured range.
       * @param sigma_      Range noise standard deviation.
       */
      RangeMeasurement(size_t pose_index, size_t Lmk_index,
                       Scalar range_, Scalar sigma_)
          : i(pose_index), j(Lmk_index), range(range_), sigma(sigma_) {}

      /**
       * @brief Stream operator for easy printing of the measurement.
       *
       * @param os           Output stream.
       * @param measurement  Measurement to print.
       * @return             Reference to the output stream.
       */
      inline friend std::ostream& operator<<(std::ostream& os,
                                             const RangeMeasurement& measurement) {
        os << "i: " << measurement.i << std::endl;
        os << "j: " << measurement.j << std::endl;
        os << "range: " << measurement.range << std::endl;
        os << "sigma: " << measurement.sigma << std::endl;
        return os;
      }
  };

/** @brief Typedef for a vector of RangeMeasurement structs. */
typedef std::vector<RangeMeasurement> rangeMeasurements_t;

/**
 * @brief A struct representing a relative measurement of a landmark from a pose.
 *
 * Stores the vector offset and precision of the landmark observation.
 */
  struct RelativeLandmarkMeasurement {
      /** @brief Index of the pose. */
      char ci;
      size_t i;

      /** @brief Index of the landmark. */
      char cj;
      size_t j;

      /** @brief Measured position of the landmark relative to the pose. */
      Vector l;

      /** @brief Precision of the landmark measurement. */
      Scalar nu;

      /** @brief Default constructor. Leaves fields uninitialized. */
      RelativeLandmarkMeasurement() {}

      /**
       * @brief Constructs a RelativeLandmarkMeasurement with given data.
       *
       * @param pose      Index of the pose.
       * @param landmark  Index of the landmark.
       * @param relative_position  Landmark position relative to the pose.
       * @param precision Precision of the measurement.
       */
      RelativeLandmarkMeasurement(size_t pose, size_t landmark,
                                  const Vector& relative_position,
                                  Scalar precision)
          : i(pose), j(landmark), l(relative_position), nu(precision) {}

      /**
       * @brief Stream operator for easy printing of the measurement.
       *
       * @param os           Output stream.
       * @param measurement  Measurement to print.
       * @return             Reference to the output stream.
       */
      inline friend std::ostream& operator<<(
          std::ostream& os, const RelativeLandmarkMeasurement& measurement) {
        os << "i: " << measurement.i << std::endl;
        os << "j: " << measurement.j << std::endl;
        os << "l: " << std::endl << measurement.l << std::endl;
        os << "nu: " << measurement.nu << std::endl;
        return os;
      }
  };

/** @brief Typedef for a vector of RelativeLandmarkMeasurement structs. */
typedef std::vector<RelativeLandmarkMeasurement> landmarkMeasurements_t;

/**
 * @brief Aggregates all measurement types parsed from data files.
 *
 * Contains vectors of pose, range, and landmark measurements, along
 * with counters for the number of poses, landmarks, and ranges.
 */
    struct Measurement {
          /** @brief Vector of relative pose measurements. */
          measurements_t poseMeasurements;

          /** @brief Vector of range measurements. */
          rangeMeasurements_t rangeMeasurements;

          /** @brief Vector of relative landmark measurements. */
          landmarkMeasurements_t landmarkMeasurements;

          /** @brief Number of poses present in the data. */
          size_t num_poses = 0;

          /** @brief Number of landmarks present in the data. */
          size_t num_landmarks = 0;

          /** @brief Number of range measurements present in the data. */
          size_t num_ranges = 0;

          /** @brief Default constructor. */
          Measurement() = default;
    };

    }  // namespace DataParser
}  // namespace gtsam


#endif //STIEFELMANIFOLDEXAMPLE_RELATIVEPOSEMEASUREMENT_H
