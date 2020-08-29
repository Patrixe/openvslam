//
// Created by Patrick Liedtke on 15.09.20.
//

#ifndef OPENVSLAM_SEGMENTED_ORB_EXTRACTOR_H
#define OPENVSLAM_SEGMENTED_ORB_EXTRACTOR_H

#include <openvslam/segmentation_config.h>
#include "orb_extractor.h"

namespace openvslam {
    namespace feature {
        class segmented_orb_extractor : public orb_extractor {
        public:
            segmented_orb_extractor() = delete;

            segmented_orb_extractor(const orb_params &orb_params, const std::shared_ptr<segmentation_config> &seg_cfg);

            segmented_orb_extractor(const unsigned int max_num_keypts, const float scale_factor,
                                    const unsigned int num_levels, const unsigned int ini_fast_thr,
                                    const unsigned int min_fast_thr,
                                    const std::shared_ptr<segmentation_config> &seg_cfg,
                                    const std::vector<std::vector<float>> &mask_rects = {});

            //! Extract keypoints and each descriptor of them
            void extract(const cv::_InputArray& in_image, const cv::Mat& seg_img, const cv::_InputArray& in_image_mask,
                         std::vector<cv::KeyPoint>& keypts, const cv::_OutputArray& out_descriptors);

        protected:
            std::shared_ptr<segmentation_config> seg_cfg;

            //! Compute fast keypoints for cells in each image pyramid, but respect semantic information
            void compute_fast_keypoints(std::vector<std::vector<cv::KeyPoint>> &all_keypts,
                                                const cv::Mat &seg_img,
                                                const cv::Mat &mask);

            void filter_by_segmentation(std::vector<cv::KeyPoint> &keypts_in_cell,
                                        const cv::Mat &segmentation_information, float scale_factor,
                                        int offset_x, int offset_y);
        };
    }
}

#endif //OPENVSLAM_SEGMENTED_ORB_EXTRACTOR_H
