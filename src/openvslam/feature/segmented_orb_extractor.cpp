//
// Created by Patrick Liedtke on 15.09.20.
//

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "segmented_orb_extractor.h"

namespace openvslam {
    namespace feature {
        segmented_orb_extractor::segmented_orb_extractor(const unsigned int max_num_keypts, const float scale_factor,
                                                         const unsigned int num_levels, const unsigned int ini_fast_thr,
                                                         const unsigned int min_fast_thr,
                                                         const std::shared_ptr<segmentation_config> &seg_cfg,
                                                         const std::vector<std::vector<float>> &mask_rects)
                : orb_extractor(max_num_keypts, scale_factor, num_levels, ini_fast_thr, min_fast_thr, mask_rects),
                  seg_cfg(seg_cfg) {
        }

        segmented_orb_extractor::segmented_orb_extractor(const orb_params &orb_params,
                                                         const std::shared_ptr<segmentation_config> &seg_cfg)
                : orb_extractor(orb_params), seg_cfg(seg_cfg) {
        }

        void segmented_orb_extractor::compute_fast_keypoints(
                std::vector<data::keypoint_container> &all_keypts,
                const cv::Mat &seg_img, const cv::Mat &mask) {
            all_keypts.resize(orb_params_.num_levels_);

            // An anonymous function which checks mask(image or rectangle)
            auto is_in_mask = [&mask](const unsigned int y, const unsigned int x, const float scale_factor) {
                return mask.at<unsigned char>(y * scale_factor, x * scale_factor) == 0;
            };

            constexpr unsigned int overlap = 6;
            constexpr unsigned int cell_size = 64;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
            for (unsigned int level = 0; level < orb_params_.num_levels_; ++level) {
                const float scale_factor = scale_factors_.at(level);

                constexpr unsigned int min_border_x = orb_patch_radius_;
                constexpr unsigned int min_border_y = orb_patch_radius_;
                const unsigned int max_border_x = image_pyramid_.at(level).cols - orb_patch_radius_;
                const unsigned int max_border_y = image_pyramid_.at(level).rows - orb_patch_radius_;

                const unsigned int width = max_border_x - min_border_x;
                const unsigned int height = max_border_y - min_border_y;

                const unsigned int num_cols = std::ceil(width / cell_size) + 1;
                const unsigned int num_rows = std::ceil(height / cell_size) + 1;

                data::keypoint_container keypts_to_distribute;
                keypts_to_distribute.reserve(orb_params_.max_num_keypts_ * 10);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
                for (unsigned int i = 0; i < num_rows; ++i) {
                    // defines overlapping "stripes" along the image which are searched one after another
                    const unsigned int min_y = min_border_y + i * cell_size;
                    if (max_border_y - overlap <= min_y) {
                        continue;
                    }
                    unsigned int max_y = min_y + cell_size + overlap;
                    if (max_border_y < max_y) {
                        max_y = max_border_y;
                    }

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
                    for (unsigned int j = 0; j < num_cols; ++j) {
                        const unsigned int min_x = min_border_x + j * cell_size;
                        if (max_border_x - overlap <= min_x) {
                            continue;
                        }
                        unsigned int max_x = min_x + cell_size + overlap;
                        if (max_border_x < max_x) {
                            max_x = max_border_x;
                        }

                        // Pass FAST computation if one of the corners of a patch is in the mask
                        if (!mask.empty()) {
                            if (is_in_mask(min_y, min_x, scale_factor) || is_in_mask(max_y, min_x, scale_factor)
                                || is_in_mask(min_y, max_x, scale_factor) || is_in_mask(max_y, max_x, scale_factor)) {
                                continue;
                            }
                        }

                        std::vector<cv::KeyPoint> keypts_in_cell;
                        data::keypoint_container seg_keypts_in_cell;
                        cv::FAST(image_pyramid_.at(level).rowRange(min_y, max_y).colRange(min_x, max_x),
                                 keypts_in_cell, orb_params_.ini_fast_thr_, true);

                        transform_to_keypoint_structure(keypts_in_cell, seg_keypts_in_cell);
                        apply_segmentation_information(seg_keypts_in_cell, seg_img, scale_factor, min_x, min_y);

                        // Re-compute FAST keypoint with reduced threshold if not enough key points were found
                        if (keypts_in_cell.empty()) {
                            cv::FAST(image_pyramid_.at(level).rowRange(min_y, max_y).colRange(min_x, max_x),
                                     keypts_in_cell, orb_params_.min_fast_thr, true);

                            if (keypts_in_cell.empty()) {
                                continue;
                            }

                            transform_to_keypoint_structure(keypts_in_cell, seg_keypts_in_cell);
                            apply_segmentation_information(seg_keypts_in_cell, seg_img, scale_factor, min_x, min_y);
                        }


                        // Collect keypoints for every scale
#ifdef USE_OPENMP
#pragma omp critical
#endif
                        {
                            for (auto &keypt : seg_keypts_in_cell) {
                                keypt.get_cv_keypoint().pt.x += j * cell_size;
                                keypt.get_cv_keypoint().pt.y += i * cell_size;
                                // Check if the keypoint is in the mask
                                if (!mask.empty() &&
                                    is_in_mask(min_border_y + keypt.get_cv_keypoint().pt.y,
                                               min_border_x + keypt.get_cv_keypoint().pt.x, scale_factor)) {
                                    continue;
                                }
                                keypts_to_distribute.push_back(keypt);
                            }
                        }
                    }
                }

                data::keypoint_container &keypts_at_level = all_keypts.at(level);
                keypts_at_level.reserve(orb_params_.max_num_keypts_);

                // Distribute keypoints via tree
                keypts_at_level = distribute_keypoints_via_tree(keypts_to_distribute,
                                                                min_border_x, max_border_x, min_border_y, max_border_y,
                                                                num_keypts_per_level_.at(level));

                // Keypoint size is patch size modified by the scale factor
                const unsigned int scaled_patch_size = fast_patch_size_ * scale_factors_.at(level);

                for (auto &keypt : keypts_at_level) {
                    // Translation correction (scale will be corrected after ORB description)
                    keypt.get_cv_keypoint().pt.x += min_border_x;
                    keypt.get_cv_keypoint().pt.y += min_border_y;
                    // Set the other information
                    keypt.get_cv_keypoint().octave = level;
                    keypt.get_cv_keypoint().size = scaled_patch_size;
                }
            }

            // Compute orientations
            for (unsigned int level = 0; level < orb_params_.num_levels_; ++level) {
                compute_orientation(image_pyramid_.at(level), all_keypts.at(level));
            }
        }

        void segmented_orb_extractor::extract(const cv::_InputArray &in_image, const cv::Mat &seg_img,
                                              const cv::_InputArray &in_image_mask, data::keypoint_container &keypts) {
            if (in_image.empty()) {
                return;
            }

            // get cv::Mat of image
            const auto image = in_image.getMat();
            assert(image.type() == CV_8UC1);

            // build image pyramid
            compute_image_pyramid(image);

            // mask initialization
            if (!mask_is_initialized_ && !orb_params_.mask_rects_.empty()) {
                create_rectangle_mask(image.cols, image.rows);
                mask_is_initialized_ = true;
            }

            std::vector<data::keypoint_container> all_keypts;

            // select mask to use
            if (!in_image_mask.empty()) {
                // Use image_mask if it is available
                const auto image_mask = in_image_mask.getMat();
                assert(image_mask.type() == CV_8UC1);
                compute_fast_keypoints(all_keypts, seg_img, image_mask);
            } else if (!rect_mask_.empty()) {
                // Use rectangle mask if it is available and image_mask is not used
                assert(rect_mask_.type() == CV_8UC1);
                compute_fast_keypoints(all_keypts, seg_img, rect_mask_);
            } else {
                // Do not use any mask if all masks are unavailable
                compute_fast_keypoints(all_keypts, seg_img, cv::Mat());
            }

            unsigned int num_keypts = 0;
            for (unsigned int level = 0; level < orb_params_.num_levels_; ++level) {
                num_keypts += all_keypts.at(level).size();
            }

            keypts.clear();
            keypts.reserve(num_keypts);

            unsigned int offset = 0;
            for (unsigned int level = 0; level < orb_params_.num_levels_; ++level) {

                auto &keypts_at_level = all_keypts.at(level);
                const auto num_keypts_at_level = keypts_at_level.size();

                if (num_keypts_at_level == 0) {
                    continue;
                }

                cv::Mat blurred_image = image_pyramid_.at(level).clone();
                cv::GaussianBlur(blurred_image, blurred_image, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
                compute_orb_descriptors(blurred_image, keypts_at_level);

                offset += num_keypts_at_level;

                correct_keypoint_scale(keypts_at_level, level);

                keypts.insert(keypts.end(), keypts_at_level.begin(), keypts_at_level.end());
            }
        }

        void segmented_orb_extractor::transform_to_keypoint_structure(std::vector<cv::KeyPoint> &keypts_in_cell,
                                                                      data::keypoint_container &seg_keypts_in_cell) {
//            seg_keypts_in_cell.reserve(keypts_in_cell.size());
            for (auto &point : keypts_in_cell) {
                seg_keypts_in_cell.push_back(data::keypoint(point));
            }
        }

        data::keypoint_container segmented_orb_extractor::distribute_keypoints_via_tree(
                const data::keypoint_container &keypts_to_distribute,
                const int min_x, const int max_x, const int min_y, const int max_y,
                const unsigned int num_keypts) const {
            auto nodes = initialize_nodes(keypts_to_distribute, min_x, max_x, min_y, max_y);

            // Forkable leaf nodes list
            // The pool is used when a forking makes nodes more than a limited number
            std::vector<std::pair<int, orb_extractor_node *>> leaf_nodes_pool;
            leaf_nodes_pool.reserve(nodes.size() * 10);

            // A flag denotes if enough keypoints have been distributed
            bool is_filled = false;

            while (true) {
                const unsigned int prev_size = nodes.size();

                auto iter = nodes.begin();
                leaf_nodes_pool.clear();

                // Fork node and remove the old one from nodes
                while (iter != nodes.end()) {
                    if (iter->is_leaf_node_) {
                        iter++;
                        continue;
                    }

                    // Divide node and assign to the leaf node pool
                    const auto child_nodes = iter->divide_node();
                    assign_child_nodes(child_nodes, nodes, leaf_nodes_pool);
                    // Remove the old node
                    iter = nodes.erase(iter);
                }

                // Stop iteration when the number of nodes is over the designated size or new node is not generated
                if (num_keypts <= nodes.size() || nodes.size() == prev_size) {
                    is_filled = true;
                    break;
                }

                // If all nodes number is more than limit, keeping nodes are selected by next step
                if (num_keypts < nodes.size() + leaf_nodes_pool.size()) {
                    is_filled = false;
                    break;
                }
            }

            while (!is_filled) {
                // Select nodes so that keypoint number is just same as designeted number
                const unsigned int prev_size = nodes.size();

                auto prev_leaf_nodes_pool = leaf_nodes_pool;
                leaf_nodes_pool.clear();

                // Sort by number of keypoints in the patch of each leaf node
                std::sort(prev_leaf_nodes_pool.rbegin(), prev_leaf_nodes_pool.rend());
                // Do processes from the node which has much more keypoints
                for (const auto &prev_leaf_node : prev_leaf_nodes_pool) {
                    // Divide node and assign to the leaf node pool
                    const auto child_nodes = prev_leaf_node.second->divide_node();
                    assign_child_nodes(child_nodes, nodes, leaf_nodes_pool);
                    // Remove the old node
                    nodes.erase(prev_leaf_node.second->iter_);

                    if (num_keypts <= nodes.size()) {
                        is_filled = true;
                        break;
                    }
                }

                // Stop dividing if the number of nodes is reached to the limit or there are no dividable nodes
                if (is_filled || num_keypts <= nodes.size() || nodes.size() == prev_size) {
                    is_filled = true;
                    break;
                }
            }

            return find_keypoints_with_max_response(nodes);
        }

        void segmented_orb_extractor::apply_segmentation_information(data::keypoint_container &keypts_in_cell,
                                                                        const cv::Mat &segmentation_information,
                                                                        float scale_factor, int offset_x, int offset_y) {
            for (auto &keypoint : keypts_in_cell) {
                if (!this->seg_cfg->allowed_for_landmark(
                        segmentation_information.at<uchar>((keypoint.get_cv_keypoint().pt.y + offset_y) * scale_factor,
                                                           (keypoint.get_cv_keypoint().pt.x + offset_x) * scale_factor))) {
                    keypoint.set_applicable_for_slam(false);
                }
            }
        }
    }
}
