//
// Created by Patrick Liedtke on 28.08.20.
//

#ifndef OPENVSLAM_SEGMENTATION_FRAME_H
#define OPENVSLAM_SEGMENTATION_FRAME_H

#include <openvslam/segmentation_config.h>
#include "frame.h"

namespace openvslam {
    namespace data {
        class segmentation_frame : public frame {
        public:
            struct keypoint {
            public:
                const cv::KeyPoint &keyPoint;
                int seg_class = -1;

                keypoint(const cv::KeyPoint &kPoint, const int seg_class) : keyPoint(kPoint), seg_class(seg_class) {};
            };

            std::vector<keypoint> segmented_keypoints;
            segmentation_config *seg_cfg;

            /**
             * Constructor for monocular frame
             * @param img_gray
             * @param seg_img
             * @param timestamp
             * @param extractor
             * @param bow_vocab
             * @param camera
             * @param depth_thr
             * @param mask
             */
            segmentation_frame(const cv::Mat &img_gray, const cv::Mat &seg_img, segmentation_config *seg_cfg,
                               const double timestamp, feature::orb_extractor *extractor,
                               bow_vocabulary *bow_vocab, camera::base *camera, const float depth_thr,
                               const cv::Mat &mask = cv::Mat{});

            /**
             * Constructor for stereo frame
             * @param left_img_gray
             * @param right_img_gray
             * @param seg_img
             * @param timestamp
             * @param extractor_left
             * @param extractor_right
             * @param bow_vocab
             * @param camera
             * @param depth_thr
             * @param mask
             */
            segmentation_frame(const cv::Mat &left_img_gray, const cv::Mat &right_img_gray, const cv::Mat &left_seg_img,
                               const cv::Mat &right_seg_img, const double timestamp,
                               feature::orb_extractor *extractor_left, feature::orb_extractor *extractor_right,
                               bow_vocabulary *bow_vocab, camera::base *camera, const float depth_thr,
                               const cv::Mat &mask = cv::Mat{});

            /**
             * Constructor for RGBD frame
             * @param img_gray
             * @param seg_img
             * @param img_depth
             * @param timestamp
             * @param extractor
             * @param bow_vocab
             * @param camera
             * @param depth_thr
             * @param mask
             */
            segmentation_frame(const cv::Mat &img_gray, const cv::Mat &seg_img, const cv::Mat &img_depth,
                               const double timestamp,
                               feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                               camera::base *camera, const float depth_thr,
                               const cv::Mat &mask = cv::Mat{});

            /**
             * Applies the class information of the given matrix to the key points detected by an orb extractor.
             * The key points coordinates are used to extract the class information stored in segmentation_information.
             * @param segmentation_information matrix that contains a number (interpreted as class number) matching the image that is processed within this frame.
             * @return a vector containing all key points found, each of enriched with class information.
             */
            void merge_segmentation_information(const cv::Mat &segmentation_information);

            void extract_orb(const cv::Mat& img, const cv::Mat& seg_img, const cv::Mat& mask, const image_side& img_side = image_side::Left);

            void filter_by_segmentation(const cv::Mat &segmentation_information);

            void filter_by_segmentation_stereo(const cv::Mat &left_seg_img, const cv::Mat &right_seg_img);
        };
    }
}

#endif //OPENVSLAM_SEGMENTATION_FRAME_H
