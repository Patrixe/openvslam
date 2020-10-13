//
// Created by Patrick Liedtke on 12.10.20.
//

#ifndef OPENVSLAM_SEGMENTATION_FRAME_PUBLISHER_H
#define OPENVSLAM_SEGMENTATION_FRAME_PUBLISHER_H

#include <opencv2/core/mat.hpp>
#include <openvslam/tracking_module.h>
#include "segmentation_tracking_module.h"

namespace openvslam {
    class segmentation_tracking_module;

    namespace data {
        class map_database;
    }

    namespace publish {
        class segmentation_frame_publisher {
        public:
            segmentation_frame_publisher(const std::shared_ptr<config> &cfg, data::map_database *map_db,
                                         unsigned int img_width = 1024);

            void update(segmentation_tracking_module *tracker);

        protected:
            // colors (BGR)
            const cv::Scalar mapping_color_{0, 255, 255};
            const cv::Scalar localization_color_{255, 255, 0};

            //! config
            std::shared_ptr<config> cfg_;
            //! map database
            data::map_database* map_db_;
            //! maximum size of output images
            const int img_width_;

            // -------------------------------------------
            //! mutex to access variables below
            std::mutex mtx_;

            //! raw img
            cv::Mat img_;
            //! tracking state
            tracker_state_t tracking_state_;

            //! initial keypoints
            std::vector<cv::KeyPoint> init_keypts_;
            //! matching between initial frame and current frame
            std::vector<int> init_matches_;

            //! current keypoints
            std::vector<cv::KeyPoint> curr_keypts_;

            //! elapsed time for tracking
            double elapsed_ms_ = 0.0;

            //! mapping module status
            bool mapping_is_enabled_;

            //! tracking flag for each current keypoint
            std::vector<bool> is_tracked_;

            cv::Mat draw_frame(const bool draw_text);

            unsigned int draw_initial_points(cv::Mat &img, const std::vector<cv::KeyPoint> &init_keypts,
                                             const std::vector<int> &init_matches,
                                             const std::vector<cv::KeyPoint> &curr_keypts,
                                             const float mag) const;

            void draw_info_text(cv::Mat &img, const openvslam::tracker_state_t tracking_state,
                                const unsigned int num_tracked,
                                const double elapsed_ms, const bool mapping_is_enabled) const;

            unsigned int draw_tracked_points(cv::Mat &img, const std::__1::vector<cv::KeyPoint> &curr_keypts,
                                             const std::__1::vector<bool> &is_tracked, const bool mapping_is_enabled,
                                             const float mag) const;
        };
    }
}

#endif //OPENVSLAM_SEGMENTATION_FRAME_PUBLISHER_H
