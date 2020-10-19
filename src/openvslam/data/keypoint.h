//
// Created by Patrick Liedtke on 13.10.20.
//

#include <opencv2/core.hpp>
#include <openvslam/type.h>

#ifndef OPENVSLAM_KEYPOINT_H
#define OPENVSLAM_KEYPOINT_H
namespace openvslam {
    namespace data {
        struct keypoint {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            // key: class, value: point coordinates
//                const std::map<int, std::tuple<int>> pixel_per_class;



            explicit keypoint(cv::KeyPoint kPoint) : cvKeyPoint(std::move(kPoint)) {
                orb_descriptor = std::make_shared<std::array<uchar, 32>>();
            };

            keypoint() {
                orb_descriptor = std::make_shared<std::array<uchar, 32>>();
            }

            bool is_applicable_for_slam() const {
                return applicable_for_slam;
            }

            void set_applicable_for_slam(bool applicable) {
                applicable_for_slam = applicable;
            }

            cv::KeyPoint& get_cv_keypoint() {
                return cvKeyPoint;
            }

            const cv::KeyPoint& get_cv_keypoint() const {
                return cvKeyPoint;
            }

            Eigen::Vector3d get_bearing() const {
                return bearing;
            }

            void set_bearing(const Eigen::Vector3d bearing) {
                this->bearing = bearing;
            }

            std::shared_ptr<std::array<uchar, 32>> get_orb_descriptor() {
                return orb_descriptor;
            }

            void set_orb_descriptor(std::shared_ptr<std::array<uchar, 32>> orbDescriptor) {
                orb_descriptor = orbDescriptor;
            }

            // TODO testen
            cv::Mat get_orb_descriptor_as_cv_mat() const {
                // size is fixed in orb_extractor
                return cv::Mat(1, 32, CV_8UC1, orb_descriptor.get());
            }
        protected:
            cv::KeyPoint cvKeyPoint;
            int seg_class = -1;
            bool applicable_for_slam = true;
            //! bearing vector
            Eigen::Vector3d bearing;
            // orb descriptor
            std::shared_ptr<std::array<uchar, 32>> orb_descriptor;
        };

        class keypoint_container : public std::vector<keypoint> {
            using std::vector<keypoint>::vector;
        public:
            /**
             * This is intended for read only access. Adding a new keypoint will not change the keypoint_container
             * @return all keypoints that are stored in this container.
             */
            std::vector<cv::KeyPoint> get_all_cv_keypoints() const;

            /**
             * This is intended for read only access. Adding a new keypoint will not change the keypoint_container
             * @return all keypoints that are stored in this container and which are flagged as being applicable for algorithms
             *          which aim to compute a map position or a pose, etc.
             */
            std::vector<cv::KeyPoint> get_slam_applicable_cv_keypoints() const;

            std::vector<cv::KeyPoint> get_slam_forbidden_cv_keypoints() const;

            std::vector<keypoint> get_slam_applicable_keypoints() const;

            std::vector<keypoint> get_slam_forbidden_keypoints() const;

            eigen_alloc_vector<Vec3_t> get_slam_applicable_bearings() const;

            eigen_alloc_vector<Vec3_t> get_slam_forbidden_bearings() const;
        };
    }
}
#endif //OPENVSLAM_KEYPOINT_H
