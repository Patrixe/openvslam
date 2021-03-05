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

            if (keypts_.empty()) {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);
            assert(keypts_.size() == undist_keypts_.size());

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
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

            if (keypts_.empty()) {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Estimate depth with stereo match
            match::stereo stereo_matcher(extractor_left->image_pyramid_, extractor_right_->image_pyramid_,
                                         keypts_, keypts_right_,
                                         scale_factors_, inv_scale_factors_,
                                         camera->focal_x_baseline_, camera_->true_baseline_);
            stereo_matcher.compute();

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_);

            // Assign all the keypoints into grid
            openvslam::data::assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
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
            if (keypts_.empty()) {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Calculate disparity from depth
            compute_stereo_from_depth(img_depth);

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
        }

        void segmentation_frame::extract_orb(const cv::Mat &img, const cv::Mat &seg_img, const cv::Mat &mask,
                                             const frame::image_side &img_side) {
            switch (img_side) {
                case image_side::Left: {
                    dynamic_cast<feature::segmented_orb_extractor *>(extractor_)->extract(img, seg_img, mask,
                                                                                          keypts_);
                    break;
                }
                case image_side::Right: {
                    dynamic_cast<feature::segmented_orb_extractor *>(extractor_right_)->extract(img, seg_img, mask,
                                                                                                keypts_right_);
                    break;
                }
            }
        }
    }
}
