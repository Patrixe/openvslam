//
// Created by Patrick Liedtke on 28.08.20.
//

#include "segmentation_frame.h"
#include "openvslam/data/common.h"
#include "openvslam/data/landmark.h"
#include "openvslam/feature/orb_extractor.h"
#include "openvslam/feature/segmented_orb_extractor.h"
#include <openvslam/match/stereo.h>
#include <thread>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace openvslam {
    namespace data {
        segmentation_frame::segmentation_frame(const cv::Mat &img_gray, const cv::Mat &seg_img, const double timestamp,
                                               feature::orb_extractor *extractor,
                                               ::openvslam::data::bow_vocabulary *bow_vocab, camera::base *camera,
                                               const float depth_thr,
                                               const cv::Mat &mask)
                : frame(timestamp, extractor, nullptr, bow_vocab, camera, depth_thr) {
            // Get ORB scale
            update_orb_info();

            // Extract ORB feature
            extract_orb(img_gray, seg_img, mask);

            num_keypts_ = keypts_.size();
            if (keypts_.empty()) {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Ignore stereo parameters
            stereo_x_right_ = std::vector<float>(num_keypts_, -1);
            depths_ = std::vector<float>(num_keypts_, -1);

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<openvslam::data::landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_.get_all_cv_keypoints(), keypt_indices_in_cells_);
        }

        // stereo image
        segmentation_frame::segmentation_frame(const cv::Mat &left_img_gray, const cv::Mat &right_img_gray,
                                               const cv::Mat &left_seg_img, const cv::Mat &right_seg_img,
                                               const double timestamp,
                                               feature::orb_extractor *extractor_left,
                                               feature::orb_extractor *extractor_right,
                                               openvslam::data::bow_vocabulary *bow_vocab, camera::base *camera,
                                               const float depth_thr,
                                               const cv::Mat &mask)
                : frame(timestamp, extractor_left, extractor_right, bow_vocab, camera, depth_thr) {
            // Get ORB scale
            update_orb_info();

            // Extract ORB feature
            std::thread thread_left(&segmentation_frame::extract_orb, this, left_img_gray, left_seg_img, mask,
                                    image_side::Left);
            std::thread thread_right(&segmentation_frame::extract_orb, this, right_img_gray, right_seg_img, mask,
                                     image_side::Right);
            thread_left.join();
            thread_right.join();

            num_keypts_ = keypts_.size();
            if (keypts_.empty()) {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Estimate depth with stereo match
            match::stereo stereo_matcher(extractor_left->image_pyramid_, extractor_right_->image_pyramid_,
                                         keypts_.get_all_cv_keypoints(), keypts_right_.get_all_cv_keypoints(), descriptors_, descriptors_right_,
                                         scale_factors_, inv_scale_factors_,
                                         camera->focal_x_baseline_, camera_->true_baseline_);
            stereo_matcher.compute(stereo_x_right_, depths_);

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<openvslam::data::landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            openvslam::data::assign_keypoints_to_grid(camera_, undist_keypts_.get_all_cv_keypoints(), keypt_indices_in_cells_);
        }

        // depth image
        segmentation_frame::segmentation_frame(const cv::Mat &img_gray, const cv::Mat &seg_img,
                                               const cv::Mat &img_depth, const double timestamp,
                                               feature::orb_extractor *extractor,
                                               ::openvslam::data::bow_vocabulary *bow_vocab,
                                               camera::base *camera, const float depth_thr,
                                               const cv::Mat &mask)
                : frame(timestamp, extractor, nullptr, bow_vocab, camera, depth_thr) {
            // Get ORB scale
            update_orb_info();

            // Extract ORB feature
            extract_orb(img_gray, seg_img, mask);
            num_keypts_ = keypts_.size();
            if (keypts_.empty()) {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Calculate disparity from depth
            compute_stereo_from_depth(img_depth);

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<openvslam::data::landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_.get_all_cv_keypoints(), keypt_indices_in_cells_);
        }
//
//        // TODO pali: the stereo problem remains, is this handled somewhere else in the code?
//        void segmentation_frame::filter_by_segmentation_stereo(const cv::Mat &left_seg_img,
//                                                               const cv::Mat &right_seg_img) {
//            auto left_seg_filter = [left_seg_img, this](cv::KeyPoint &kp) {
//                return !this->seg_cfg->allowed_for_landmark(left_seg_img.at<cv::Vec3b>(kp.pt)[0]);
//            };
//            auto right_seg_filter = [right_seg_img, this](cv::KeyPoint &kp) {
//                return !this->seg_cfg->allowed_for_landmark(right_seg_img.at<cv::Vec3b>(kp.pt)[0]);
//            };
//
//            keypts_.erase(std::remove_if(keypts_.begin(), keypts_.end(), left_seg_filter), keypts_.end());
//            keypts_right_.erase(std::remove_if(keypts_right_.begin(), keypts_right_.end(), right_seg_filter),
//                                keypts_right_.end());
//        }

        void segmentation_frame::extract_orb(const cv::Mat &img, const cv::Mat &seg_img, const cv::Mat &mask,
                                             const frame::image_side &img_side) {
            switch (img_side) {
                case image_side::Left: {
                    dynamic_cast<feature::segmented_orb_extractor *>(extractor_)->extract(img, seg_img, mask,
                                                                                          keypts_,descriptors_);
                    break;
                }
                case image_side::Right: {
                    dynamic_cast<feature::segmented_orb_extractor *>(extractor_right_)->extract(img, seg_img, mask,
                                                                                                keypts_,descriptors_right_);
                    break;
                }
            }
        }
    }
}
