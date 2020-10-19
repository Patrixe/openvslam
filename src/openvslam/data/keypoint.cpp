//
// Created by Patrick Liedtke on 15.10.20.
//
#include "keypoint.h"

namespace openvslam {
    namespace data {
        std::vector<cv::KeyPoint> keypoint_container::get_all_cv_keypoints() const {
            std::vector<cv::KeyPoint> keypoints;
            keypoints.reserve(this->size());
            for (auto& keypoint : *this) {
                keypoints.emplace_back(keypoint.get_cv_keypoint());
            }

            return keypoints;
        }

        std::vector<cv::KeyPoint> keypoint_container::get_slam_applicable_cv_keypoints() const {
            std::vector<cv::KeyPoint> keypoints;
            keypoints.reserve(this->size());
            for (auto& keypoint : *this) {
                if (keypoint.is_applicable_for_slam()) {
                    keypoints.emplace_back(keypoint.get_cv_keypoint());
                }
            }

            return keypoints;
        }

        std::vector<cv::KeyPoint> keypoint_container::get_slam_forbidden_cv_keypoints() const {
            std::vector<cv::KeyPoint> keypoints;
            keypoints.reserve(this->size());
            for (auto& keypoint : *this) {
                if (!keypoint.is_applicable_for_slam()) {
                    keypoints.emplace_back(keypoint.get_cv_keypoint());
                }
            }

            return keypoints;
        }

        std::vector<keypoint> keypoint_container::get_slam_applicable_keypoints() const {
            std::vector<keypoint> keypoints;
            keypoints.reserve(this->size());
            for (auto& keypoint : *this) {
                if (keypoint.is_applicable_for_slam()) {
                    keypoints.emplace_back(keypoint);
                }
            }

            return keypoints;
        }
        std::vector<keypoint> keypoint_container::get_slam_forbidden_keypoints() const {
            std::vector<keypoint> keypoints;
            keypoints.reserve(this->size());
            for (auto& keypoint : *this) {
                if (!keypoint.is_applicable_for_slam()) {
                    keypoints.emplace_back(keypoint);
                }
            }

            return keypoints;
        }

        eigen_alloc_vector<Vec3_t> keypoint_container::get_slam_applicable_bearings() const {
            eigen_alloc_vector<Vec3_t> bearings;
            bearings.reserve(this->size());
            for (auto& keypoint : *this) {
                if (keypoint.is_applicable_for_slam()) {
                    bearings.emplace_back(keypoint.get_bearing());
                }
            }

            return bearings;
        }

        eigen_alloc_vector<Vec3_t> keypoint_container::get_slam_forbidden_bearings() const {
            eigen_alloc_vector<Vec3_t> bearings;
            bearings.reserve(this->size());
            for (auto& keypoint : *this) {
                if (!keypoint.is_applicable_for_slam()) {
                    bearings.emplace_back(keypoint.get_bearing());
                }
            }

            return bearings;
        }
    }
}