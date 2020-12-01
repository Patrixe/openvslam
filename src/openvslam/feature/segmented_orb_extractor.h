//
// Created by Patrick Liedtke on 15.09.20.
//

#ifndef OPENVSLAM_SEGMENTED_ORB_EXTRACTOR_H
#define OPENVSLAM_SEGMENTED_ORB_EXTRACTOR_H

#include <openvslam/segmentation/segmentation_config.h>
#include <openvslam/data/segmentation_frame.h>
#include "openvslam/feature/orb_extractor.h"

namespace openvslam {
    namespace feature {
        class segmented_orb_extractor : public orb_extractor {
        public:
            segmented_orb_extractor() = delete;

            segmented_orb_extractor(const orb_params &orb_params, const std::shared_ptr<segmentation_config> &seg_cfg);

            segmented_orb_extractor(unsigned int max_num_keypts, float scale_factor,
                                    unsigned int num_levels, unsigned int ini_fast_thr,
                                    unsigned int min_fast_thr,
                                    const std::shared_ptr<segmentation_config> &seg_cfg,
                                    const std::vector<std::vector<float>> &mask_rects = {});

            //! Extract keypoints and each descriptor of them
            void extract(const cv::_InputArray &in_image, const cv::Mat &seg_img,
                         const cv::_InputArray &in_image_mask, data::keypoint_container &keypts);

        protected:
            std::shared_ptr<segmentation_config> seg_cfg;

            //! Compute fast keypoints for cells in each image pyramid, but respect semantic information
            void compute_fast_keypoints(
                    std::vector<data::keypoint_container> &all_keypts,
                    const cv::Mat &seg_img,
                    const cv::Mat &mask);

            void apply_segmentation_information(
                    data::keypoint_container &keypts_in_cell,
                    const cv::Mat &segmentation_information, float scale_factor,
                    int offset_x, int offset_y);

            data::keypoint_container distribute_keypoints_via_tree(
                    const data::keypoint_container &keypts_to_distribute,
                    int min_x, int max_x, int min_y, int max_y, unsigned int num_keypts) const override;

            static void transform_to_keypoint_structure(std::vector<cv::KeyPoint> &keypts_in_cell,
                                                 data::keypoint_container &seg_keypts_in_cell);

            void extract_fast_keypoint_classes(const cv::Mat &segmentation_information, float scale_factor,
                                               int offset_x,
                                               int offset_y, const std::pair<const int, data::keypoint> &keypoint,
                                               std::map<int, int, std::greater<int>> &class_count) const;
        };
    }
}

#endif //OPENVSLAM_SEGMENTED_ORB_EXTRACTOR_H
