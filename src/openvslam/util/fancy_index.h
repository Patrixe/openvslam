#ifndef OPENVSLAM_UTIL_FANCY_INDEX_H
#define OPENVSLAM_UTIL_FANCY_INDEX_H

#include "openvslam/type.h"
#include <openvslam/data/keypoint.h>
#include <openvslam/data/landmark.h>

#include <vector>
#include <type_traits>

namespace openvslam {
namespace util {

template<typename T, typename U>
std::vector<T> resample_by_indices(const std::vector<T>& elements, const std::vector<U>& indices) {
    static_assert(std::is_integral<U>(), "the element type of indices must be integer");

    std::vector<T> resampled;
    resampled.reserve(elements.size());
    for (const auto idx : indices) {
        resampled.push_back(elements.at(idx));
    }

    return resampled;
}

template<typename T, typename U>
eigen_alloc_vector<T> resample_by_indices(const eigen_alloc_vector<T>& elements, const std::vector<U>& indices) {
    static_assert(std::is_integral<U>(), "the element type of indices must be integer");

    eigen_alloc_vector<T> resampled;
    resampled.reserve(elements.size());
    for (const auto idx : indices) {
        resampled.push_back(elements.at(idx));
    }

    return resampled;
}

template<typename T>
std::vector<T> resample_by_indices(const std::vector<T>& elements, const std::vector<bool>& indices) {
    assert(elements.size() == indices.size());

    std::vector<T> resampled;
    resampled.reserve(elements.size());
    for (unsigned int idx = 0; idx < elements.size(); ++idx) {
        if (indices.at(idx)) {
            resampled.push_back(elements.at(idx));
        }
    }

    return resampled;
}

template<typename T>
eigen_alloc_vector<T> resample_by_indices(const eigen_alloc_vector<T>& elements, const std::vector<bool>& indices) {
    assert(elements.size() == indices.size());

    eigen_alloc_vector<T> resampled;
    resampled.reserve(elements.size());
    for (unsigned int idx = 0; idx < elements.size(); ++idx) {
        if (indices.at(idx)) {
            resampled.push_back(elements.at(idx));
        }
    }

    return resampled;
}

std::vector<data::landmark*> resample_by_indices(const std::map<int, data::landmark*> &elements, const std::vector<unsigned int>& indices);
std::vector<cv::KeyPoint> resample_by_indices(const std::map<int, data::keypoint> &elements, const std::vector<unsigned int>& indices);
eigen_alloc_vector<Vec3_t> resample_by_indices(const eigen_alloc_map<int, Vec3_t> &elements, const std::vector<unsigned int>& indices);

} // namespace util
} // namespace openvslam

#endif // OPENVSLAM_UTIL_FANCY_INDEX_H
