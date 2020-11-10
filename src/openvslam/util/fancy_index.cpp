//
// Created by Patrick Liedtke on 29.10.20.
//
#include "openvslam/util/fancy_index.h"

namespace openvslam {
    namespace util {
        std::vector<data::landmark*> resample_by_indices(const std::map<int, data::landmark*> &elements, const std::vector<unsigned int>& indices) {
            std::vector<data::landmark*> resampled;
            resampled.reserve(elements.size());
            for (auto idx : elements) {
                if (std::find(indices.begin(), indices.end(), idx.first) != indices.end()) {
                    resampled.push_back(idx.second);
                }
            }

            return resampled;
        }

        std::vector<cv::KeyPoint> resample_by_indices(const std::map<int, data::keypoint> &elements, const std::vector<unsigned int>& indices) {
            std::vector<cv::KeyPoint> resampled;
            resampled.reserve(elements.size());
            for (auto index : indices) {
                resampled.push_back(elements.at(index).get_cv_keypoint());
            }

            return resampled;
        }

        eigen_alloc_vector<Vec3_t> resample_by_indices(const eigen_alloc_map<int, Vec3_t> &elements, const std::vector<unsigned int>& indices) {
            eigen_alloc_vector<Vec3_t> resampled;
            resampled.reserve(elements.size());
            for (auto index : indices) {
                resampled.push_back(elements.at(index));
            }

            return resampled;
        }
    }
}