//
// Created by Patrick Liedtke on 28.08.20.
//

#ifndef OPENVSLAM_SEGMENTATION_SYSTEM_H
#define OPENVSLAM_SEGMENTATION_SYSTEM_H

#include "openvslam/system.h"
#include "segmentation_config.h"
#include <spdlog/spdlog.h>

namespace openvslam {
    class config;

    class tracking_module;

    class mapping_module;

    class global_optimization_module;

    class audit_exporter;

    namespace camera {
        class base;
    } // namespace camera

    namespace data {
        class camera_database;

        class map_database;

        class bow_database;
    } // namespace data

    namespace publish {
        class map_publisher;

        class segmentation_frame_publisher;
    } // namespace publish

    class segmentation_system : public system {
    public:
        //! extended constructor for segmentation add in.
        segmentation_system(const std::shared_ptr<config> &cfg, const std::string &vocab_file_path,
                            std::shared_ptr<segmentation_config> seg_cfg);

        Mat44_t
        feed_monocular_frame(const cv::Mat &img, const cv::Mat &seg_img, double timestamp, const cv::Mat &mask);

        Mat44_t feed_stereo_frame(const cv::Mat &left_img, const cv::Mat &right_img, const cv::Mat &left_seg_img,
                                  const cv::Mat &right_seg_img, const double timestamp, const cv::Mat &mask);

        ~segmentation_system() override;

    private:
        audit_exporter *audit_exporter;
    };
}

#endif //OPENVSLAM_SEGMENTATION_SYSTEM_H
