//
// Created by Patrick Liedtke on 13.10.20.
//

#include <opencv2/core.hpp>

#ifndef OPENVSLAM_KEYPOINT_H
#define OPENVSLAM_KEYPOINT_H
namespace openvslam {
    namespace data {
        struct keypoint {
        public:
            // key: class, value: point coordinates
//                const std::map<int, std::tuple<int>> pixel_per_class;

            explicit keypoint(cv::KeyPoint kPoint) : cvKeyPoint(std::move(kPoint)) {};

            keypoint() = default;

            keypoint(const keypoint &kp) = default;

//            void operator=(const cv::KeyPoint& kp) {
//                this->cvKeyPoint = kp;
//            }

            bool get_applicable_for_slam() {
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

        protected:
            cv::KeyPoint cvKeyPoint;
            int seg_class = -1;
            bool applicable_for_slam = true;
        };

        class keypoint_container : public std::vector<keypoint> {
            using std::vector<keypoint>::vector;
        public:
            /**
             * This is intended for read only access. Adding a new keypoint will not change the keypoint_container
             * @return
             */
            std::vector<cv::KeyPoint> get_all_cv_keypoints() const {
                std::vector<cv::KeyPoint> keypoints;
                keypoints.reserve(this->size());
                for (auto& keypoint : *this) {
                    keypoints.emplace_back(keypoint.get_cv_keypoint());
                }

                return keypoints;
            }
        };
    }
}
#endif //OPENVSLAM_KEYPOINT_H
