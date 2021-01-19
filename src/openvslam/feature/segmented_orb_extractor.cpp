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
                                keypt.second.get_cv_keypoint().pt.x += j * cell_size;
                                keypt.second.get_cv_keypoint().pt.y += i * cell_size;
                                // Check if the keypoint is in the mask
                                if (!mask.empty() &&
                                    is_in_mask(min_border_y + keypt.second.get_cv_keypoint().pt.y,
                                               min_border_x + keypt.second.get_cv_keypoint().pt.x, scale_factor)) {
                                    continue;
                                }
                                keypts_to_distribute.insert(keypt);
                            }
                        }
                    }
                }

                data::keypoint_container &keypts_at_level = all_keypts.at(level);

                // Distribute keypoints via tree
                keypts_at_level = distribute_keypoints_via_tree(keypts_to_distribute,
                                                                min_border_x, max_border_x, min_border_y, max_border_y,
                                                                num_keypts_per_level_.at(level));

                // Keypoint size is patch size modified by the scale factor
                const unsigned int scaled_patch_size = fast_patch_size_ * scale_factors_.at(level);

                for (auto &keypt : keypts_at_level) {
                    // Translation correction (scale will be corrected after ORB description)
                    keypt.second.get_cv_keypoint().pt.x += min_border_x;
                    keypt.second.get_cv_keypoint().pt.y += min_border_y;
                    // Set the other information
                    keypt.second.get_cv_keypoint().octave = level;
                    keypt.second.get_cv_keypoint().size = scaled_patch_size;
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

                keypts.insert(keypts_at_level.begin(), keypts_at_level.end());
            }
        }

        void segmented_orb_extractor::transform_to_keypoint_structure(std::vector<cv::KeyPoint> &keypts_in_cell,
                                                                      data::keypoint_container &seg_keypts_in_cell) {
            for (auto &point : keypts_in_cell) {
                const data::keypoint &keypoint = data::keypoint(point);
                seg_keypts_in_cell.insert(std::pair<int, data::keypoint>(keypoint.get_id(), keypoint));
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

        std::list<orb_extractor_node> segmented_orb_extractor::initialize_nodes(
                const data::keypoint_container &keypts_to_distribute,
                const int min_x, const int max_x, const int min_y, const int max_y) const {
            // The aspect ratio of the target area for keypoint detection
            const auto ratio = static_cast<double>(max_x - min_x) / (max_y - min_y);
            // The width and height of the patches allocated to the initial node
            double delta_x, delta_y;
            // The number of columns or rows
            unsigned int num_x_grid, num_y_grid;

            if (ratio > 1) {
                // If the aspect ratio is greater than 1, the patches are made in a horizontal direction
                num_x_grid = std::round(ratio);
                num_y_grid = 1;
                delta_x = static_cast<double>(max_x - min_x) / num_x_grid;
                delta_y = max_y - min_y;
            } else {
                // If the aspect ratio is equal to or less than 1, the patches are made in a vertical direction
                num_x_grid = 1;
                num_y_grid = std::round(1 / ratio);
                delta_x = max_x - min_y;
                delta_y = static_cast<double>(max_y - min_y) / num_y_grid;
            }

            // The number of the initial nodes
            const unsigned int num_initial_nodes = num_x_grid * num_y_grid;

            // A list of node
            std::list<orb_extractor_node> nodes;

            // Initial node objects
            std::vector<orb_extractor_node *> initial_nodes;
            initial_nodes.resize(num_initial_nodes);

            // Create initial node substances
            for (unsigned int i = 0; i < num_initial_nodes; ++i) {
                orb_extractor_node node;

                // x / y index of the node's patch in the grid
                const unsigned int ix = i % num_x_grid;
                const unsigned int iy = i / num_x_grid;

                node.pt_begin_ = cv::Point2i(delta_x * ix, delta_y * iy);
                node.pt_end_ = cv::Point2i(delta_x * (ix + 1), delta_y * (iy + 1));

                nodes.push_back(node);
                initial_nodes.at(i) = &nodes.back();
            }

            // Assign all keypoints to initial nodes which own keypoint's position
            for (const auto &keypt : keypts_to_distribute) {
                if (!keypt.second.is_applicable_for_slam()) {
                    continue;
                }

                // x / y index of the patch where the keypt is placed
                const unsigned int ix = keypt.second.get_cv_keypoint().pt.x / delta_x;
                const unsigned int iy = keypt.second.get_cv_keypoint().pt.y / delta_y;

                const unsigned int node_idx = ix + iy * num_x_grid;
                initial_nodes.at(node_idx)->keypts_.insert(keypt);
            }

            auto iter = nodes.begin();
            while (iter != nodes.end()) {
                // Remove empty nodes
                if (iter->keypts_.empty()) {
                    iter = nodes.erase(iter);
                    continue;
                }
                // Set the leaf node flag if the node has only one keypoint
                iter->is_leaf_node_ = (iter->keypts_.size() == 1);
                iter++;
            }

            return nodes;
        }

        void segmented_orb_extractor::apply_segmentation_information(data::keypoint_container &keypts_in_cell,
                                                                        const cv::Mat &segmentation_information,
                                                                        float scale_factor, int offset_x, int offset_y) {
            if (seg_cfg->get_segmentation_assignment_mode() == 0) {
                for (auto &keypoint : keypts_in_cell) {
                    const auto &seg_class = segmentation_information.at<uchar>(
                            (keypoint.second.get_cv_keypoint().pt.y + offset_y) * scale_factor,
                            (keypoint.second.get_cv_keypoint().pt.x + offset_x) * scale_factor);

                    keypoint.second.set_segmentation_class(seg_class);

                    if (!this->seg_cfg->allowed_for_landmark(seg_class)) {
                        keypoint.second.set_applicable_for_slam(false);
                    }
                }

                return;
            }

            if (seg_cfg->get_segmentation_assignment_mode() == 1) {
                for (auto &keypoint : keypts_in_cell) {
                    // query pixels all around the center of the feature for their class. Assuming 9_16 configuration of FAST detector.
                    std::map<int, int, std::greater<int>> class_count;
                    extract_fast_keypoint_classes(segmentation_information, scale_factor, offset_x, offset_y, keypoint,
                                                  class_count);

                    int seg_class = -1;
                    for (auto segmentation_class : class_count) {
                        // variable threshold to filter out noise
                        if (segmentation_class.second > 1) {
                            seg_class = segmentation_class.first;
                            break;
                        }
                    }

                    keypoint.second.set_segmentation_class(seg_class);
                    if (!this->seg_cfg->allowed_for_landmark(seg_class)) {
                        keypoint.second.set_applicable_for_slam(false);
                    }
                }
            }
        }

        void segmented_orb_extractor::extract_fast_keypoint_classes(const cv::Mat &segmentation_information,
                                                                    float scale_factor,
                                                                    int offset_x, int offset_y,
                                                                    const std::pair<const int, data::keypoint> &keypoint,
                                                                    std::map<int, int, std::greater<int>> &class_count) const {
            // left three
            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y - 1) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x - 3) * scale_factor)]++;

            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x - 3) * scale_factor)]++;

            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y + 1) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x - 3) * scale_factor)]++;

            // right three
            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y - 1) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x + 3) * scale_factor)]++;

            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x + 3) * scale_factor)]++;

            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y + 1) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x + 3) * scale_factor)]++;

            // bottom three
            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y - 3) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x - 1) * scale_factor)]++;

            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y - 3) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x) * scale_factor)]++;

            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y - 3) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x + 1) * scale_factor)]++;

            // top three
            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y + 3) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x - 1) * scale_factor)]++;

            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y + 3) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x) * scale_factor)]++;

            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y + 3) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x + 1) * scale_factor)]++;

            // diagonal cases
            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y - 2) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x - 2) * scale_factor)]++;

            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y - 2) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x + 2) * scale_factor)]++;

            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y + 2) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x - 2) * scale_factor)]++;

            class_count[segmentation_information.at<uchar>(
                    (keypoint.second.get_cv_keypoint().pt.y + offset_y + 2) * scale_factor,
                    (keypoint.second.get_cv_keypoint().pt.x + offset_x + 2) * scale_factor)]++;

        }
    }
}
