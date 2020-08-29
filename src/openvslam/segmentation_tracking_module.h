//
// Created by Patrick Liedtke on 27.08.20.
//

#ifndef OPENVSLAM_SEGMENTATION_TRACKING_MODULE_H
#define OPENVSLAM_SEGMENTATION_TRACKING_MODULE_H

#include "tracking_module.h"
#include "openvslam/config.h"
#include "openvslam/feature/orb_extractor.h"
#include "segmentation_config.h"

namespace openvslam {

    class segmentation_tracking_module : public tracking_module {
    public:
        std::shared_ptr<segmentation_config> seg_cfg;

        segmentation_tracking_module(std::shared_ptr<config> cfg, system *system, data::map_database *map_db,
                                     data::bow_vocabulary *bow_vocab, data::bow_database *bow_db,
                                     const std::shared_ptr<segmentation_config> &seg_cfg);

        //! Track a monocular frame
        //! (NOTE: distorted images are acceptable if calibrated)
        Mat44_t track_monocular_image(const cv::Mat &img, const cv::Mat &seg_img, double timestamp,
                                      const cv::Mat &mask = cv::Mat{});

        //! Track a stereo frame
        //! (Note: Left and Right images must be stereo-rectified)
        Mat44_t track_stereo_image(const cv::Mat &left_img_rect, const cv::Mat &right_img_rect,
                                   const cv::Mat &left_seg_img, const cv::Mat &right_seg_img, double timestamp,
                                   const cv::Mat &mask = cv::Mat{});

        //! Track an RGBD frame
        //! (Note: RGB and Depth images must be aligned)
        Mat44_t track_RGBD_image(const cv::Mat &img, const cv::Mat &seg_img, const cv::Mat &depthmap, double timestamp,
                                 const cv::Mat &mask = cv::Mat{});
    };

}
#endif //OPENVSLAM_SEGMENTATION_TRACKING_MODULE_H
