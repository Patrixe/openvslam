//
// Created by Patrick Liedtke on 28.08.20.
//

#ifndef OPENVSLAM_SEGMENTATION_FRAME_H
#define OPENVSLAM_SEGMENTATION_FRAME_H

#include <openvslam/segmentation/segmentation_config.h>

#include <utility>
#include "openvslam/data/frame.h"
#include "keypoint.h"

namespace openvslam {
        namespace data {
        class segmentation_frame : public frame {
            public:
                segmentation_frame() = default;

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
                segmentation_frame(const cv::Mat &img_gray, const cv::Mat &seg_img,
                                   double timestamp, feature::orb_extractor *extractor,
                                   openvslam::data::bow_vocabulary *bow_vocab, camera::base *camera, float depth_thr,
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
                segmentation_frame(const cv::Mat &left_img_gray, const cv::Mat &right_img_gray,
                                   const cv::Mat &left_seg_img,
                                   const cv::Mat &right_seg_img, double timestamp,
                                   feature::orb_extractor *extractor_left, feature::orb_extractor *extractor_right,
                                   ::openvslam::data::bow_vocabulary *bow_vocab, camera::base *camera, float depth_thr,
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
                                   double timestamp,
                                   feature::orb_extractor *extractor, ::openvslam::data::bow_vocabulary *bow_vocab,
                                   camera::base *camera, float depth_thr,
                                   const cv::Mat &mask = cv::Mat{});

                void extract_orb(const cv::Mat &img, const cv::Mat &seg_img, const cv::Mat &mask,
                                 const image_side &img_side = image_side::Left);
            };
    }
}

#endif //OPENVSLAM_SEGMENTATION_FRAME_H
