//
// Created by Patrick Liedtke on 27.08.20.
//

#include <openvslam/util/image_converter.h>
#include <openvslam/data/segmentation_frame.h>
#include <openvslam/feature/segmented_orb_extractor.h>
#include "segmentation_tracking_module.h"

namespace openvslam {
    openvslam::segmentation_tracking_module::segmentation_tracking_module(const std::shared_ptr<config> cfg,
                                                                          system *system, data::map_database *map_db,
                                                                          data::bow_vocabulary *bow_vocab,
                                                                          data::bow_database *bow_db,
                                                                          const std::shared_ptr<segmentation_config> &seg_cfg)
            : tracking_module(cfg, system, map_db, bow_vocab, bow_db), seg_cfg(seg_cfg) {
        if (seg_cfg) {
            extractor_left_ = new feature::segmented_orb_extractor(cfg->orb_params_, seg_cfg);
            if (camera_->setup_type_ == camera::setup_type_t::Monocular) {
                ini_extractor_left_ = new feature::segmented_orb_extractor(cfg_->orb_params_, seg_cfg);
                ini_extractor_left_->set_max_num_keypoints(ini_extractor_left_->get_max_num_keypoints() * 2);
            }
            if (camera_->setup_type_ == camera::setup_type_t::Stereo) {
                extractor_right_ = new feature::segmented_orb_extractor(cfg_->orb_params_, seg_cfg);
            }
        }
    }

    Mat44_t segmentation_tracking_module::track_monocular_image(const cv::Mat &img, const cv::Mat &seg_img,
                                                                const double timestamp, const cv::Mat &mask) {
        const auto start = std::chrono::system_clock::now();

        // color conversion
        img_gray_ = img;
        util::convert_to_grayscale(img_gray_, camera_->color_order_);

        // create current frame object
        if (tracking_state_ == tracker_state_t::NotInitialized || tracking_state_ == tracker_state_t::Initializing) {
            curr_frm_ = data::segmentation_frame(img_gray_, seg_img, seg_cfg.get(), timestamp, ini_extractor_left_,
                                                 bow_vocab_, camera_, cfg_->true_depth_thr_, mask);
        } else {
            curr_frm_ = data::segmentation_frame(img_gray_, seg_img, seg_cfg.get(), timestamp, extractor_left_,
                                                 bow_vocab_, camera_, cfg_->true_depth_thr_, mask);
        }

        track();

        const auto end = std::chrono::system_clock::now();
        elapsed_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        return curr_frm_.cam_pose_cw_;
    }

    Mat44_t
    segmentation_tracking_module::track_stereo_image(const cv::Mat &left_img_rect, const cv::Mat &right_img_rect,
                                                     const cv::Mat &left_seg_img, const cv::Mat &right_seg_img,
                                                     const double timestamp, const cv::Mat &mask) {
        const auto start = std::chrono::system_clock::now();

        // color conversion
        img_gray_ = left_img_rect;
        cv::Mat right_img_gray = right_img_rect;
        util::convert_to_grayscale(img_gray_, camera_->color_order_);
        util::convert_to_grayscale(right_img_gray, camera_->color_order_);

        // create current frame object
        curr_frm_ = data::segmentation_frame(img_gray_, right_img_gray, left_seg_img, right_seg_img, timestamp,
                                             extractor_left_, extractor_right_, bow_vocab_, camera_,
                                             cfg_->true_depth_thr_, mask);

        track();

        const auto end = std::chrono::system_clock::now();
        elapsed_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        return curr_frm_.cam_pose_cw_;
    }

    Mat44_t
    segmentation_tracking_module::track_RGBD_image(const cv::Mat &img, const cv::Mat &seg_img, const cv::Mat &depthmap,
                                                   const double timestamp, const cv::Mat &mask) {
        const auto start = std::chrono::system_clock::now();

        // color and depth scale conversion
        img_gray_ = img;
        cv::Mat img_depth = depthmap;
        util::convert_to_grayscale(img_gray_, camera_->color_order_);
        util::convert_to_true_depth(img_depth, cfg_->depthmap_factor_);

        // create current frame object
        curr_frm_ = data::frame(img_gray_, img_depth, timestamp, extractor_left_, bow_vocab_, camera_,
                                cfg_->true_depth_thr_, mask);

        track();

        const auto end = std::chrono::system_clock::now();
        elapsed_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        return curr_frm_.cam_pose_cw_;
    }
}