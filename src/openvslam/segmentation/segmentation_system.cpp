//
// Created by Patrick Liedtke on 28.08.20.
//

#include "segmentation_system.h"
#include "openvslam/config.h"
#include "openvslam/data/camera_database.h"
#include "openvslam/data/map_database.h"
#include "openvslam/data/bow_database.h"
#include "openvslam/publish/map_publisher.h"
#include "openvslam/publish/frame_publisher.h"
#include "openvslam/mapping_module.h"
#include "openvslam/global_optimization_module.h"

namespace openvslam {
    segmentation_system::segmentation_system(const std::shared_ptr<config> &cfg, const std::string &vocab_file_path,
                                             const std::shared_ptr<segmentation_config> seg_cfg) :system(cfg, vocab_file_path) {
        // frame and map publisher
        frame_publisher_ = std::shared_ptr<publish::frame_publisher>(new publish::frame_publisher(cfg_, map_db_));
        map_publisher_ = std::shared_ptr<publish::map_publisher>(new publish::map_publisher(cfg_, map_db_));

        // tracking module
        tracker_ = new tracking_module(cfg_, this, map_db_, bow_vocab_, bow_db_, seg_cfg);
        // mapping module
        mapper_ = new mapping_module(map_db_, camera_->setup_type_ == camera::setup_type_t::Monocular);
        // global optimization module
        global_optimizer_ = new global_optimization_module(map_db_, bow_db_, bow_vocab_,
                                                           camera_->setup_type_ != camera::setup_type_t::Monocular);

        // connect modules each other
        tracker_->set_mapping_module(mapper_);
        tracker_->set_global_optimization_module(global_optimizer_);
        mapper_->set_global_optimization_module(global_optimizer_);
        global_optimizer_->set_mapping_module(mapper_);
    }

    Mat44_t
    segmentation_system::feed_monocular_frame(const cv::Mat &img, const cv::Mat &seg_img, const double timestamp,
                                              const cv::Mat &mask) {
        assert(camera_->setup_type_ == camera::setup_type_t::Monocular);

        check_reset_request();

        Mat44_t cam_pose_cw = this->tracker_->track_monocular_image_with_segmentation(img, seg_img, timestamp, mask);

        frame_publisher_->update(tracker_);
        if (tracker_->is_tracking()) {
            map_publisher_->set_current_cam_pose(cam_pose_cw);
        }

        return cam_pose_cw;
    }

    Mat44_t segmentation_system::feed_stereo_frame(const cv::Mat &left_img, const cv::Mat &right_img,
                                                   const cv::Mat &left_seg_img, const cv::Mat &right_seg_img,
                                                   const double timestamp, const cv::Mat &mask) {
        assert(camera_->setup_type_ == camera::setup_type_t::Stereo);

        check_reset_request();

        const Mat44_t cam_pose_cw = this->tracker_->track_stereo_image_with_segmentation(left_img, right_img, left_seg_img, right_seg_img, timestamp, mask);

        frame_publisher_->update(tracker_);
        if (tracker_->is_tracking()) {
            map_publisher_->set_current_cam_pose(cam_pose_cw);
        }

        return cam_pose_cw;
    }
}