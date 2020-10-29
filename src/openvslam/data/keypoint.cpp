//
// Created by Patrick Liedtke on 15.10.20.
//
#include "keypoint.h"

namespace openvslam {
    namespace data {
        int keypoint::id_counter = 0;

        keypoint::keypoint(cv::KeyPoint kPoint) : cvKeyPoint(std::move(kPoint)) {
            orb_descriptor = std::make_shared<std::array<uchar, 32>>();
            // not thread safe
            point_id = id_counter++;
        };

        keypoint::keypoint() {
            orb_descriptor = std::make_shared<std::array<uchar, 32>>();
            // not thread sage
            point_id = id_counter++;
        }

        bool keypoint::is_applicable_for_slam() const {
            return applicable_for_slam;
        }

        void keypoint::set_applicable_for_slam(bool applicable) {
            applicable_for_slam = applicable;
        }

        cv::KeyPoint& keypoint::get_cv_keypoint() {
            return cvKeyPoint;
        }

        const cv::KeyPoint& keypoint::get_cv_keypoint() const {
            return cvKeyPoint;
        }

        Eigen::Vector3d keypoint::get_bearing() const {
            return bearing;
        }

        void keypoint::set_bearing(const Eigen::Vector3d& bearing) {
            this->bearing = bearing;
        }

        std::shared_ptr<std::array<uchar, 32>> keypoint::get_orb_descriptor() {
            return orb_descriptor;
        }

        void keypoint::set_orb_descriptor(std::shared_ptr<std::array<uchar, 32>> orbDescriptor) {
            orb_descriptor = orbDescriptor;
        }

        // TODO testen
        cv::Mat keypoint::get_orb_descriptor_as_cv_mat() const {
            // size is fixed in orb_extractor
            return cv::Mat(1, 32, CV_8UC1, orb_descriptor.get());
        }

        int keypoint::get_id() const {
            return point_id;
        }

        float keypoint::get_depth() const {
            return depth;
        }

        void keypoint::set_depth(float depth) {
            this->depth = depth;
        }

        float keypoint::get_stereo_x_offset() const {
            return stereo_x_offset;
        }

        void keypoint::set_stereo_x_offset(float offset) {
            stereo_x_offset = offset;
        }

        void keypoint::reset_id_counter() {
            keypoint::id_counter = 0;
        }

        std::vector<cv::KeyPoint> keypoint_container::get_all_cv_keypoints() const {
            std::vector<cv::KeyPoint> keypoints;
            keypoints.reserve(this->size());
            for (auto& keypoint : *this) {
                keypoints.emplace_back(keypoint.second.get_cv_keypoint());
            }

            return keypoints;
        }

        std::vector<cv::KeyPoint> keypoint_container::get_slam_applicable_cv_keypoints() const {
            std::vector<cv::KeyPoint> keypoints;
            keypoints.reserve(this->size());
            for (auto& keypoint : *this) {
                if (keypoint.second.is_applicable_for_slam()) {
                    keypoints.emplace_back(keypoint.second.get_cv_keypoint());
                }
            }

            return keypoints;
        }

        std::vector<cv::KeyPoint> keypoint_container::get_slam_forbidden_cv_keypoints() const {
            std::vector<cv::KeyPoint> keypoints;
            keypoints.reserve(this->size());
            for (auto& keypoint : *this) {
                if (!keypoint.second.is_applicable_for_slam()) {
                    keypoints.emplace_back(keypoint.second.get_cv_keypoint());
                }
            }

            return keypoints;
        }

        std::map<int, keypoint> keypoint_container::get_slam_applicable_keypoints() const {
            std::map<int, keypoint> keypoints;
            for (auto& keypoint : *this) {
                if (keypoint.second.is_applicable_for_slam()) {
                    keypoints.insert(keypoint);
                }
            }

            return keypoints;
        }
        std::map<int, keypoint> keypoint_container::get_slam_forbidden_keypoints() const {
            std::map<int, keypoint> keypoints;
            for (auto& keypoint : *this) {
                if (!keypoint.second.is_applicable_for_slam()) {
                    keypoints.insert(keypoint);
                }
            }

            return keypoints;
        }

        eigen_alloc_map<int, Vec3_t> keypoint_container::get_slam_applicable_bearings() const {
            eigen_alloc_map<int, Vec3_t> bearings;
            for (auto& keypoint : *this) {
                if (keypoint.second.is_applicable_for_slam()) {
                    bearings.insert(std::pair<int, Vec3_t >(keypoint.first, keypoint.second.get_bearing()));
                }
            }

            return bearings;
        }

        eigen_alloc_map<int, Vec3_t> keypoint_container::get_slam_forbidden_bearings() const {
            eigen_alloc_map<int, Vec3_t> bearings;
            for (auto& keypoint : *this) {
                if (!keypoint.second.is_applicable_for_slam()) {
                    bearings.insert(std::pair<int, Vec3_t >(keypoint.first, keypoint.second.get_bearing()));
                }
            }

            return bearings;
        }

        eigen_alloc_map<int, Vec3_t> keypoint_container::get_all_bearings() const {
            eigen_alloc_map<int, Vec3_t> bearings;
            for (auto& keypoint : *this) {
                bearings.insert(std::pair<int, Vec3_t >(keypoint.first, keypoint.second.get_bearing()));
            }

            return bearings;
        }
    }
}