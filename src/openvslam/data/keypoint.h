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
            static void reset_id_counter();

            void set_segmentation_class(const int seg_class);

        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            // key: class, value: point coordinates
//                const std::map<int, std::tuple<int>> pixel_per_class;

            keypoint();

            explicit keypoint(cv::KeyPoint kPoint);

            bool is_applicable_for_slam() const;

            void set_applicable_for_slam(bool applicable);

            cv::KeyPoint& get_cv_keypoint();

            const cv::KeyPoint& get_cv_keypoint() const;

            Eigen::Vector3d get_bearing() const;

            void set_bearing(const Eigen::Vector3d& bearing);

            uchar* get_orb_descriptor_pointer();

            // TODO testen
            cv::Mat get_orb_descriptor_as_cv_mat() const;

            int get_id() const;

            float get_depth() const;

            void set_depth(float);

            float get_stereo_x_offset() const;

            void set_stereo_x_offset(float);
        protected:
            int point_id = -1;
            cv::KeyPoint cvKeyPoint;
            int seg_class = -1;
            bool applicable_for_slam = true;
            //! bearing vector
            Eigen::Vector3d bearing;
            cv::Mat orb_descriptor = cv::Mat::zeros(1, 32, CV_8UC1);
            float depth = 0;
            float stereo_x_offset = 0;

        private:
            static int id_counter;
        };

        class keypoint_container : public std::map<int, keypoint> {
            using std::map<int, keypoint>::map;
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

            std::map<int, keypoint> get_slam_applicable_keypoints() const;

            std::map<int, keypoint> get_slam_forbidden_keypoints() const;

            eigen_alloc_map<int, Vec3_t> get_slam_applicable_bearings() const;

            eigen_alloc_map<int, Vec3_t> get_slam_forbidden_bearings() const;

            eigen_alloc_map<int, Vec3_t> get_all_bearings() const;
        };
    }
}
#endif //OPENVSLAM_KEYPOINT_H
