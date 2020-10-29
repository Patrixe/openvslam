#include "openvslam/data/frame.h"
#include "openvslam/data/keypoint.h"
#include "openvslam/match/area.h"
#include "openvslam/match/angle_checker.h"

namespace openvslam {
    namespace match {

        unsigned int area::match_in_consistent_area(data::frame &frm_2, std::map<int, cv::Point2f> &prev_matched_pts,
                                                    // maps: point_id of point in frame a -> pair(point of frameA, point of frameB)
                                                    std::map<int, std::pair<data::keypoint, data::keypoint>> &found_matches,
                                                    int margin,
                                                    const std::map<int, data::keypoint> &frame_1_slam_cv_points,
                                                    const std::map<int, data::keypoint> &frame_2_slam_cv_points) {
            if (frame_1_slam_cv_points.empty() || frame_2_slam_cv_points.empty()) {
                return 0;
            }

            found_matches.clear();

            angle_checker<int> angle_checker;
            std::map<int, unsigned int> matched_dists_in_frm_2;
            for (auto point : frame_2_slam_cv_points) {
                matched_dists_in_frm_2.insert(std::pair<int, unsigned int>(point.first, MAX_HAMMING_DIST));
            }

            std::map<int, int> matched_indices_1_in_frm_2;
            for (auto point : frame_2_slam_cv_points) {
                matched_indices_1_in_frm_2.insert(std::pair<int, int>(point.first, -1));
            }

            for (const auto &frame_a_point : frame_1_slam_cv_points) {
                const auto scale_level_1 = frame_a_point.second.get_cv_keypoint().octave;

                // 第0スケールの特徴点のみを用いる
                if (0 < scale_level_1) {
                    continue;
                }

                // 一つ前にマッチングした特徴点周辺のcellの特徴点を持ってくる
                // if the initialization process takes more than two frames, this is used to "track" the movement of a
                // point from the initial frame to the frame thats processed now. The gap between the correct matches would otherwise increase every frame.
                std::vector<std::reference_wrapper<const data::keypoint>> neighbouring_keypoints = frm_2.get_keypoints_in_cell(
                        prev_matched_pts.at(frame_a_point.first).x, prev_matched_pts.at(frame_a_point.first).y,
                        margin, scale_level_1, scale_level_1);

                if (neighbouring_keypoints.empty()) {
                    continue;
                }

                const auto &frame_a_point_descriptor = frame_a_point.second.get_orb_descriptor_as_cv_mat();

                unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;
                std::reference_wrapper<const data::keypoint> best_matching_keypoint_of_frame_b = neighbouring_keypoints[0];

                for (std::reference_wrapper<const data::keypoint> frame_b_point_to_be_checked : neighbouring_keypoints) {
                    if (!frame_b_point_to_be_checked.get().is_applicable_for_slam()) {
                        continue;
                    }

                    const auto &desc_2 = frame_b_point_to_be_checked.get().get_orb_descriptor_as_cv_mat();
                    const auto hamm_dist = compute_descriptor_distance_32(frame_a_point_descriptor, desc_2);

                    // すでにマッチした点のほうが近ければスルーする
                    // check whether this point of frame 2 has a lower distance to another point from a previous iteration
                    if (matched_dists_in_frm_2.at(frame_b_point_to_be_checked.get().get_id()) <= hamm_dist) {
                        continue;
                    }

                    if (hamm_dist < best_hamm_dist) {
                        second_best_hamm_dist = best_hamm_dist;
                        best_hamm_dist = hamm_dist;
                        best_matching_keypoint_of_frame_b = frame_b_point_to_be_checked;
                    } else if (hamm_dist < second_best_hamm_dist) {
                        second_best_hamm_dist = hamm_dist;
                    }
                }

                if (HAMMING_DIST_THR_LOW < best_hamm_dist) {
                    continue;
                }

                // ratio test
                if (second_best_hamm_dist * lowe_ratio_ < static_cast<float>(best_hamm_dist)) {
                    continue;
                }

                // best_idx_2に対応する点がすでに存在する場合(=prev_idx_1)は，
                // 上書きするため対応するmatched_indices_2_in_frm_1.at(prev_idx_1)の対応情報を削除する必要がある
                // (matched_indices_1_in_frm_2.at(best_idx_2)は上書きするので削除する必要がない)
                // check if the best fitting point of the second frame has already been assigned to another point in an earlier iteration
                // TODO pali: Shouldn't this be decided by comparing the distances?
                const auto prev_idx_1 = matched_indices_1_in_frm_2.at(best_matching_keypoint_of_frame_b.get().get_id());
                if (0 <= prev_idx_1) {
                    found_matches.erase(prev_idx_1);
                }

                // 互いの対応情報を記録する
                found_matches.insert(std::pair<int, std::pair<data::keypoint, data::keypoint>> (
                        frame_a_point.first,
                        std::pair<data::keypoint, data::keypoint>(frame_a_point.second, best_matching_keypoint_of_frame_b.get()))
                );
                matched_indices_1_in_frm_2.at(best_matching_keypoint_of_frame_b.get().get_id()) = frame_a_point.first;
                matched_dists_in_frm_2.at(best_matching_keypoint_of_frame_b.get().get_id()) = best_hamm_dist;

                if (check_orientation_) {
                    const auto delta_angle = frame_a_point.second.get_cv_keypoint().angle
                                             - best_matching_keypoint_of_frame_b.get().get_cv_keypoint().angle;
                    angle_checker.append_delta_angle(delta_angle, frame_a_point.first);
                }
            }

            if (check_orientation_) {
                const auto invalid_matches = angle_checker.get_invalid_matches();
                for (const auto invalid_match_index : invalid_matches) {
                    found_matches.erase(invalid_match_index);
                }
            }

            // previous matchesを更新する
            for (auto match_entry : found_matches) {
                    // looks strange, remember the map: idA -> pair(frameA, frameB)
                    prev_matched_pts.at(match_entry.first) = match_entry.second.second.get_cv_keypoint().pt;
            }

            return found_matches.size();
        }

    } // namespace match
} // namespace openvslam
