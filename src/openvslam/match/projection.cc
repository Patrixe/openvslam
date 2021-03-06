#include <spdlog/spdlog.h>
#include "openvslam/camera/base.h"
#include "openvslam/data/frame.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/keypoint.h"
#include "openvslam/data/landmark.h"
#include "openvslam/match/projection.h"
#include "openvslam/match/angle_checker.h"

namespace openvslam {
namespace match {

unsigned int projection::match_frame_and_landmarks(data::frame& frm, const std::vector<data::landmark*>& local_landmarks, const float margin) const {
    unsigned int num_matches = 0;

    for (auto local_lm : local_landmarks) {
        if (!local_lm->is_observable_in_tracking_) {
            continue;
        }
        if (local_lm->will_be_erased()) {
            continue;
        }

        const auto pred_scale_level = local_lm->scale_level_in_tracking_;

        // 3次元点を再投影した点が存在するcellの特徴点を取得
        // TODO pali: This might need adjustment regarding the ids that are returned.
        const auto keypoints_in_cell = frm.get_keypoints_in_cell(local_lm->reproj_in_tracking_(0), local_lm->reproj_in_tracking_(1),
                                                               margin * frm.scale_factors_.at(pred_scale_level),
                                                               pred_scale_level - 1, pred_scale_level);
        if (keypoints_in_cell.empty()) {
            continue;
        }

        const cv::Mat lm_desc = local_lm->get_descriptor();

        unsigned int best_hamm_dist = MAX_HAMMING_DIST;
        int best_scale_level = -1;
        unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;
        int second_best_scale_level = -1;
        std::reference_wrapper<const data::keypoint> best_matching_keypoint = keypoints_in_cell[0];

        // TODO pali: Check if the logic in here is coherent with the split of points (and subsequently ids which are simply counted up)
        for (const auto neighbouring_keypoint_ref : keypoints_in_cell) {
            auto neighbouring_keypoint = neighbouring_keypoint_ref.get();
            if ((frm.landmarks_.find(neighbouring_keypoint.get_id()) != frm.landmarks_.end()) &&
                frm.landmarks_.at(neighbouring_keypoint.get_id())->has_observation()) {
                continue;
            }

            if (neighbouring_keypoint.get_stereo_x_offset() > 0) {
                const auto reproj_error = std::abs(local_lm->x_right_in_tracking_ - neighbouring_keypoint.get_stereo_x_offset());
                if (margin * frm.scale_factors_.at(pred_scale_level) < reproj_error) {
                    continue;
                }
            }

            // TODO pali: check this, just quick fixed for compilation
            const cv::Mat& desc = neighbouring_keypoint.get_orb_descriptor_as_cv_mat();

            const auto dist = compute_descriptor_distance_32(lm_desc, desc);

            if (dist < best_hamm_dist) {
                second_best_hamm_dist = best_hamm_dist;
                best_hamm_dist = dist;
                second_best_scale_level = best_scale_level;
                best_scale_level = neighbouring_keypoint.get_cv_keypoint().octave;
                best_matching_keypoint = neighbouring_keypoint;
            }
            else if (dist < second_best_hamm_dist) {
                second_best_scale_level = neighbouring_keypoint.get_cv_keypoint().octave;
                second_best_hamm_dist = dist;
            }
        }

        if (best_hamm_dist <= HAMMING_DIST_THR_HIGH) {
            // lowe's ratio test
            if (best_scale_level == second_best_scale_level && best_hamm_dist > lowe_ratio_ * second_best_hamm_dist) {
                continue;
            }

            // 対応情報を追加
            frm.landmarks_[best_matching_keypoint.get().get_id()] = local_lm;
            ++num_matches;
        }
    }

    return num_matches;
}

unsigned int projection::match_current_and_last_frames(data::frame& curr_frm, const data::frame& last_frm, const float margin) const {
    unsigned int num_matches = 0;

    angle_checker<int> angle_checker;

    const Mat33_t rot_cw = curr_frm.cam_pose_cw_.block<3, 3>(0, 0);
    const Vec3_t trans_cw = curr_frm.cam_pose_cw_.block<3, 1>(0, 3);

    const Vec3_t trans_wc = -rot_cw.transpose() * trans_cw;

    const Mat33_t rot_lw = last_frm.cam_pose_cw_.block<3, 3>(0, 0);
    const Vec3_t trans_lw = last_frm.cam_pose_cw_.block<3, 1>(0, 3);

    const Vec3_t trans_lc = rot_lw * trans_wc + trans_lw;

    // monocular以外の場合は，current->lastの並進ベクトルのz成分で前進しているか判断しているかを判定する
    // z成分が正に振れている -> 前進している
    const bool assume_forward = (curr_frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
                                    ? false
                                    : trans_lc(2) > curr_frm.camera_->true_baseline_;
    // z成分が負に振れている -> 後退している
    const bool assume_backward = (curr_frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
                                     ? false
                                     : -trans_lc(2) > curr_frm.camera_->true_baseline_;

    // last frameの特徴点と対応が取れている3次元点を，current frameに再投影して対応を求める
    for (auto &lm : last_frm.landmarks_) {
        // pose optimizationでoutlierになったものとは対応を取らない
        if (lm.second->is_outlier()) {
            continue;
        }

        // グローバル基準の3次元点座標
        const Vec3_t pos_w = lm.second->get_pos_in_world();

        // 再投影して可視性を求める
        Vec2_t reproj;
        float x_right;
        const bool in_image = curr_frm.camera_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);

        // 画像外に再投影される場合はスルー
        if (!in_image) {
            continue;
        }

        // 隣接フレーム間では対応する特徴点のスケールは一定であると仮定し，探索範囲を設定
        const data::keypoint &keypoint_for_last_frame_landmark = last_frm.undist_keypts_.at(lm.first);
        const auto last_scale_level = keypoint_for_last_frame_landmark.get_cv_keypoint().octave;

        // 3次元点を再投影した点が存在するcellの特徴点を取得
        std::vector<std::reference_wrapper<const data::keypoint>> keypoints_in_cell;
        if (assume_forward) {
            keypoints_in_cell = curr_frm.get_keypoints_in_cell(reproj(0), reproj(1),
                                                     margin * curr_frm.scale_factors_.at(last_scale_level),
                                                               last_scale_level, last_frm.num_scale_levels_ - 1);
        }
        else if (assume_backward) {
            keypoints_in_cell = curr_frm.get_keypoints_in_cell(reproj(0), reproj(1),
                                                     margin * curr_frm.scale_factors_.at(last_scale_level),
                                                               0, last_scale_level);
        }
        else {
            keypoints_in_cell = curr_frm.get_keypoints_in_cell(reproj(0), reproj(1),
                                                     margin * curr_frm.scale_factors_.at(last_scale_level),
                                                     last_scale_level - 1, last_scale_level + 1);
        }

//        spdlog::debug("ProjectionMatcher: {} candidates in cell", keypoints_in_cell.size());

        if (keypoints_in_cell.empty()) {
            continue;
        }

        const auto lm_desc = lm.second->get_descriptor();

        unsigned int best_hamm_dist = MAX_HAMMING_DIST;
        std::reference_wrapper<const data::keypoint> best_matching_keypoint = keypoints_in_cell[0];

        for (const auto &neighbouring_keypoint_ref : keypoints_in_cell) {
            const auto &neighbouring_keypoint = neighbouring_keypoint_ref.get();
            if ((curr_frm.landmarks_.find(neighbouring_keypoint.get_id()) != curr_frm.landmarks_.end()) &&
                curr_frm.landmarks_[neighbouring_keypoint.get_id()]->has_observation()) {
                continue;
            }

            if (neighbouring_keypoint.get_stereo_x_offset() > 0) {
                const float reproj_error = std::fabs(x_right - neighbouring_keypoint.get_stereo_x_offset());
                if (margin * curr_frm.scale_factors_.at(last_scale_level) < reproj_error) {
                    continue;
                }
            }

            const auto& desc = neighbouring_keypoint.get_orb_descriptor_as_cv_mat();

            const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

            if (hamm_dist < best_hamm_dist) {
                best_hamm_dist = hamm_dist;
                best_matching_keypoint = neighbouring_keypoint;
            }
        }

        if (HAMMING_DIST_THR_HIGH < best_hamm_dist) {
            continue;
        }

        // 有効なmatchingとする
        curr_frm.landmarks_[best_matching_keypoint.get().get_id()] = lm.second;
        ++num_matches;

        if (check_orientation_) {
            const auto delta_angle
                = keypoint_for_last_frame_landmark.get_cv_keypoint().angle - best_matching_keypoint.get().get_cv_keypoint().angle;
            angle_checker.append_delta_angle(delta_angle, best_matching_keypoint.get().get_id());
        }
    }

    if (check_orientation_) {
        const auto invalid_matches = angle_checker.get_invalid_matches();
        for (const auto invalid_idx : invalid_matches) {
            curr_frm.landmarks_.erase(invalid_idx);
            --num_matches;
        }
    }

    return num_matches;
}

unsigned int projection::match_frame_and_keyframe(data::frame& curr_frm, data::keyframe* keyfrm, const std::set<data::landmark*>& already_matched_lms,
                                                  const float margin, const unsigned int hamm_dist_thr) const {
    unsigned int num_matches = 0;

    angle_checker<int> angle_checker;

    const Mat33_t rot_cw = curr_frm.cam_pose_cw_.block<3, 3>(0, 0);
    const Vec3_t trans_cw = curr_frm.cam_pose_cw_.block<3, 1>(0, 3);
    const Vec3_t cam_center = -rot_cw.transpose() * trans_cw;

    const auto landmarks = keyfrm->get_landmarks();

    for (unsigned int idx = 0; idx < landmarks.size(); idx++) {
        auto* lm = landmarks.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        if (already_matched_lms.count(lm)) {
            continue;
        }

        // グローバル基準の3次元点座標
        const Vec3_t pos_w = lm->get_pos_in_world();

        // 再投影して可視性を求める
        Vec2_t reproj;
        float x_right;
        const bool in_image = curr_frm.camera_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);

        // 画像外に再投影される場合はスルー
        if (!in_image) {
            continue;
        }

        // ORBスケールの範囲内であることを確認
        const Vec3_t cam_to_lm_vec = pos_w - cam_center;
        const auto cam_to_lm_dist = cam_to_lm_vec.norm();
        const auto max_cam_to_lm_dist = lm->get_max_valid_distance();
        const auto min_cam_to_lm_dist = lm->get_min_valid_distance();

        if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist) {
            continue;
        }

        // 3次元点を再投影した点が存在するcellの特徴点を取得
        const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, &curr_frm);

        const auto keypoints_in_cell = curr_frm.get_keypoints_in_cell(reproj(0), reproj(1),
                                                            margin * curr_frm.scale_factors_.at(pred_scale_level),
                                                            pred_scale_level - 1, pred_scale_level + 1);

        if (keypoints_in_cell.empty()) {
            continue;
        }

        const auto lm_desc = lm->get_descriptor();

        unsigned int best_hamm_dist = MAX_HAMMING_DIST;
        std::reference_wrapper<const data::keypoint> best_matching_keypoint = keypoints_in_cell[0];

        for (const auto neighbouring_keypoint_ref : keypoints_in_cell) {
            const auto &neighbouring_keypoint = neighbouring_keypoint_ref.get();
            if (curr_frm.landmarks_.find(neighbouring_keypoint.get_id()) != curr_frm.landmarks_.end()) {
                continue;
            }

            // TODO pali: Check this. Just quick fixed for compilation.
            const auto& desc = neighbouring_keypoint.get_orb_descriptor_as_cv_mat();

            const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

            if (hamm_dist < best_hamm_dist) {
                best_hamm_dist = hamm_dist;
                best_matching_keypoint = neighbouring_keypoint;
            }
        }

        // 有効なmatchingとする
        if (hamm_dist_thr < best_hamm_dist) {
            continue;
        }

        curr_frm.landmarks_[best_matching_keypoint.get().get_id()] = lm;
        num_matches++;

        if (check_orientation_) {
            const auto delta_angle
                = keyfrm->undist_keypts_.at(idx).get_cv_keypoint().angle - best_matching_keypoint.get().get_cv_keypoint().angle;
            angle_checker.append_delta_angle(delta_angle, best_matching_keypoint.get().get_id());
        }
    }

    if (check_orientation_) {
        // TODO pali: In all methods, this is not correct anymore. Need to use point ids, not indices
        const auto invalid_matches = angle_checker.get_invalid_matches();
        for (const auto invalid_idx : invalid_matches) {
            curr_frm.landmarks_.erase(invalid_idx);
            --num_matches;
        }
    }

    return num_matches;
}

unsigned int projection::match_by_Sim3_transform(data::keyframe* keyfrm, const Mat44_t& Sim3_cw, const std::map<int, data::landmark*>& landmarks,
                                                 std::map<int, data::landmark*>& matched_lms_in_keyfrm, const float margin) const {
    unsigned int num_matches = 0;

    // Sim3を分解してSE3にする
    const Mat33_t s_rot_cw = Sim3_cw.block<3, 3>(0, 0);
    const auto s_cw = std::sqrt(s_rot_cw.block<1, 3>(0, 0).dot(s_rot_cw.block<1, 3>(0, 0)));
    const Mat33_t rot_cw = s_rot_cw / s_cw;
    const Vec3_t trans_cw = Sim3_cw.block<3, 1>(0, 3) / s_cw;
    const Vec3_t cam_center = -rot_cw.transpose() * trans_cw;

    std::map<int, data::landmark*> already_matched(matched_lms_in_keyfrm.begin(), matched_lms_in_keyfrm.end());

    for (auto &lm : landmarks) {
        if (lm.second->will_be_erased()) {
            continue;
        }
        if (already_matched.find(lm.first) != already_matched.end()) {
            continue;
        }

        // グローバル基準の3次元点座標
        const Vec3_t pos_w = lm.second->get_pos_in_world();

        // 再投影して可視性を求める
        Vec2_t reproj;
        float x_right;
        const bool in_image = keyfrm->camera_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);

        // 画像外に再投影される場合はスルー
        if (!in_image) {
            continue;
        }

        // ORBスケールの範囲内であることを確認
        const Vec3_t cam_to_lm_vec = pos_w - cam_center;
        const auto cam_to_lm_dist = cam_to_lm_vec.norm();
        const auto max_cam_to_lm_dist = lm.second->get_max_valid_distance();
        const auto min_cam_to_lm_dist = lm.second->get_min_valid_distance();

        if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist) {
            continue;
        }

        // 3次元点の平均観測ベクトルとの角度を計算し，閾値(60deg)より大きければ破棄
        const Vec3_t obs_mean_normal = lm.second->get_obs_mean_normal();

        if (cam_to_lm_vec.dot(obs_mean_normal) < 0.5 * cam_to_lm_dist) {
            continue;
        }

        // 3次元点を再投影した点が存在するcellの特徴点を取得
        const auto pred_scale_level = lm.second->predict_scale_level(cam_to_lm_dist, keyfrm);

        const auto keypoints_in_cell = keyfrm->get_keypoints_in_cell(reproj(0), reproj(1), margin * keyfrm->scale_factors_.at(pred_scale_level));

        if (keypoints_in_cell.empty()) {
            continue;
        }

        // descriptorが最も近い特徴点を探す
        const auto lm_desc = lm.second->get_descriptor();

        unsigned int best_dist = MAX_HAMMING_DIST;
        std::reference_wrapper<const data::keypoint> best_matching_keypoint = keypoints_in_cell[0];

        for (const auto neighbouring_keypoint_ref : keypoints_in_cell) {
            auto neighbouring_keypoint = neighbouring_keypoint_ref.get();
            if (matched_lms_in_keyfrm.find(neighbouring_keypoint.get_id()) != matched_lms_in_keyfrm.end()) {
                continue;
            }

            const auto scale_level = static_cast<unsigned int>(neighbouring_keypoint.get_cv_keypoint().octave);

            // TODO: keyfrm->get_keypts_in_cell()でスケールの判断をする
            if (scale_level < pred_scale_level - 1 || pred_scale_level < scale_level) {
                continue;
            }

            const auto& desc = neighbouring_keypoint.get_orb_descriptor_as_cv_mat();

            const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

            if (hamm_dist < best_dist) {
                best_dist = hamm_dist;
                best_matching_keypoint = neighbouring_keypoint;
            }
        }

        if (HAMMING_DIST_THR_LOW < best_dist) {
            continue;
        }

        matched_lms_in_keyfrm[best_matching_keypoint.get().get_id()] = lm.second;
        ++num_matches;
    }

    return num_matches;
}

unsigned int projection::match_keyframes_mutually(data::keyframe* keyfrm_1, data::keyframe* keyfrm_2, std::map<int, data::landmark*>& matched_lms_in_keyfrm_1,
                                                  const float& s_12, const Mat33_t& rot_12, const Vec3_t& trans_12, const float margin, bool slam_applicable_only) const {
    // keyframe1の姿勢
    const Mat33_t rot_1w = keyfrm_1->get_rotation();
    const Vec3_t trans_1w = keyfrm_1->get_translation();

    // keyframe2の姿勢
    const Mat33_t rot_2w = keyfrm_2->get_rotation();
    const Vec3_t trans_2w = keyfrm_2->get_translation();

    // 与えられたカメラ間の座標系の相似変換
    const Mat33_t s_rot_12 = s_12 * rot_12;
    const Mat33_t s_rot_21 = (1.0 / s_12) * rot_12.transpose();
    const Vec3_t trans_21 = -s_rot_21 * trans_12;

    const auto landmarks_1 = keyfrm_1->get_landmarks();
    const auto landmarks_2 = keyfrm_2->get_landmarks();

    // keyframe1と2で，すでにマッチングしている特徴点があればマークしておく
    std::map<int, bool> is_already_matched_in_keyfrm_1;
    std::map<int, bool> is_already_matched_in_keyfrm_2;

    for (const auto& lm_pair : landmarks_1) {
        if (matched_lms_in_keyfrm_1.find(lm_pair.first) == matched_lms_in_keyfrm_1.end()) {
            continue;
        }

        if (slam_applicable_only && !lm_pair.second->is_applicable_for_slam()) {
            continue;
        }

        auto* lm = matched_lms_in_keyfrm_1.at(lm_pair.first);
        const auto idx_2 = lm->get_index_in_keyframe(keyfrm_2);

        is_already_matched_in_keyfrm_1[lm_pair.first] = true;
        is_already_matched_in_keyfrm_2[idx_2] = true;
    }

    std::map<int, int> matched_indices_2_in_keyfrm_1;
    std::map<int, int> matched_indices_1_in_keyfrm_2;

    // keyframe1で観測している3次元点をkeyframe2の座標系へ相似変換してから再投影し，
    // 対応している特徴点を探す
    // (world座標 -- SE3 -> keyframe1 -- Sim3 --> keyframe2)
    // s_rot_21 * (rot_1w * pos_w + trans_1w) + trans_21
    // = s_rot_21 * rot_1w * pos_w + s_rot_21 * trans_1w + trans_21
    {
        const Mat33_t s_rot_21w = s_rot_21 * rot_1w;
        const Vec3_t trans_21w = s_rot_21 * trans_1w + trans_21;
        for (const auto& lm_pair : landmarks_1) {
            auto* lm = lm_pair.second;

            if (lm->will_be_erased()) {
                continue;
            }

            if (slam_applicable_only && !lm_pair.second->is_applicable_for_slam()) {
                continue;
            }

            if (is_already_matched_in_keyfrm_1.find(lm_pair.first) != is_already_matched_in_keyfrm_1.end()) {
                continue;
            }

            // グローバル基準の3次元点座標
            const Vec3_t pos_w = lm->get_pos_in_world();
            const Vec3_t pos_2 = s_rot_21w * pos_w + trans_21w;

            // 再投影して可視性を求める
            Vec2_t reproj;
            float x_right;
            const bool in_image = keyfrm_2->camera_->reproject_to_image(s_rot_21w, trans_21w, pos_w, reproj, x_right);

            // 画像外に再投影される場合はスルー
            if (!in_image) {
                continue;
            }

            // ORBスケールの範囲内であることを確認
            const auto cam_to_lm_dist = pos_2.norm();
            const auto max_cam_to_lm_dist = lm->get_max_valid_distance();
            const auto min_cam_to_lm_dist = lm->get_min_valid_distance();

            if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist) {
                continue;
            }

            // 3次元点を再投影した点が存在するcellの特徴点を取得
            const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, keyfrm_2);

            const auto keypoints_in_cell = keyfrm_2->get_keypoints_in_cell(reproj(0), reproj(1), margin * keyfrm_2->scale_factors_.at(pred_scale_level));

            if (keypoints_in_cell.empty()) {
                continue;
            }

            // descriptorが最も近い特徴点を探す
            const auto lm_desc = lm->get_descriptor();

            unsigned int best_hamm_dist = MAX_HAMMING_DIST;
            std::reference_wrapper<const data::keypoint> best_matching_keypoint = keypoints_in_cell[0];

            for (const auto neighbouring_keypoint_ref : keypoints_in_cell) {
                auto neighbouring_keypoint = neighbouring_keypoint_ref.get();

                if (slam_applicable_only && !neighbouring_keypoint.is_applicable_for_slam()) {
                    continue;
                }

                const auto scale_level = static_cast<unsigned int>(neighbouring_keypoint.get_cv_keypoint().octave);

                // TODO: keyfrm->get_keypts_in_cell()でスケールの判断をする
                if (scale_level < pred_scale_level - 1 || pred_scale_level < scale_level) {
                    continue;
                }

                const auto& desc = neighbouring_keypoint.get_orb_descriptor_as_cv_mat();

                const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

                if (hamm_dist < best_hamm_dist) {
                    best_hamm_dist = hamm_dist;
                    best_matching_keypoint = neighbouring_keypoint;
                }
            }

            if (best_hamm_dist <= HAMMING_DIST_THR_HIGH) {
                matched_indices_2_in_keyfrm_1[lm_pair.first] = best_matching_keypoint.get().get_id();
            }
        }
    }

    // keyframe1で観測している3次元点をkeyframe2の座標系へ相似変換してから再投影し，
    // 対応している特徴点を探す
    // (world座標 -- SE3 -> keyframe2 -- Sim3 --> keyframe1)
    // s_rot_12 * (rot_2w * pos_w + trans_2w) + trans_12
    // = s_rot_12 * rot_2w * pos_w + s_rot_12 * trans_2w + trans_12
    {
        const Mat33_t s_rot_12w = s_rot_12 * rot_2w;
        const Vec3_t trans_12w = s_rot_12 * trans_2w + trans_12;
        for (const auto lm_pair_keyframe_2 : landmarks_2) {
            auto* lm = lm_pair_keyframe_2.second;
            if (lm->will_be_erased()) {
                continue;
            }

            if (is_already_matched_in_keyfrm_2.find(lm_pair_keyframe_2.first) != is_already_matched_in_keyfrm_2.end()) {
                continue;
            }

            // グローバル基準の3次元点座標
            const Vec3_t pos_w = lm->get_pos_in_world();
            const Vec3_t pos_1 = s_rot_12w * pos_w + trans_12w;

            // 再投影して可視性を求める
            Vec2_t reproj;
            float x_right;
            const bool in_image = keyfrm_2->camera_->reproject_to_image(s_rot_12w, trans_12w, pos_w, reproj, x_right);

            // 画像外に再投影される場合はスルー
            if (!in_image) {
                continue;
            }

            // ORBスケールの範囲内であることを確認
            const auto cam_to_lm_dist = pos_1.norm();
            const auto max_cam_to_lm_dist = lm->get_max_valid_distance();
            const auto min_cam_to_lm_dist = lm->get_min_valid_distance();

            if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist) {
                continue;
            }

            // 3次元点を再投影した点が存在するcellの特徴点を取得
            const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, keyfrm_1);

            const auto keypoints_in_cell = keyfrm_1->get_keypoints_in_cell(reproj(0), reproj(1), margin * keyfrm_1->scale_factors_.at(pred_scale_level));

            if (keypoints_in_cell.empty()) {
                continue;
            }

            // descriptorが最も近い特徴点を探す
            const auto lm_desc = lm->get_descriptor();

            unsigned int best_hamm_dist = MAX_HAMMING_DIST;
            std::reference_wrapper<const data::keypoint> best_matching_keypoint = keypoints_in_cell[0];

            for (const auto neighbouring_keypoint_ref : keypoints_in_cell) {
                auto neighbouring_keypoint = neighbouring_keypoint_ref.get();

                const auto scale_level = static_cast<unsigned int>(neighbouring_keypoint.get_cv_keypoint().octave);

                // TODO: keyfrm->get_keypts_in_cell()でスケールの判断をする
                if (scale_level < pred_scale_level - 1 || pred_scale_level < scale_level) {
                    continue;
                }

                const auto& desc = neighbouring_keypoint.get_orb_descriptor_as_cv_mat();

                const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

                if (hamm_dist < best_hamm_dist) {
                    best_hamm_dist = hamm_dist;
                    best_matching_keypoint = neighbouring_keypoint;
                }
            }

            if (best_hamm_dist <= HAMMING_DIST_THR_HIGH) {
                matched_indices_1_in_keyfrm_2[lm_pair_keyframe_2.first] = best_matching_keypoint.get().get_id();
            }
        }
    }

    // cross-matchのみを記録する
    unsigned int num_matches = 0;
    for (const auto &lm_1_pair : landmarks_1) {
        if (matched_indices_2_in_keyfrm_1.find(lm_1_pair.first) == matched_indices_2_in_keyfrm_1.end()) {
            continue;
        }

        int idx_2 = matched_indices_2_in_keyfrm_1.at(lm_1_pair.first);
        if (matched_indices_1_in_keyfrm_2.find(idx_2) != matched_indices_1_in_keyfrm_2.end()) {
            const auto idx_1 = matched_indices_1_in_keyfrm_2.at(idx_2);
            matched_lms_in_keyfrm_1[idx_1] = landmarks_2.at(idx_2);
            ++num_matches;
        }
    }
//    for (unsigned int i = 0; i < landmarks_1.size(); ++i) {
//        const auto idx_2 = matched_indices_2_in_keyfrm_1.at(i);
//        if (idx_2 < 0) {
//            continue;
//        }
//
//        const auto idx_1 = matched_indices_1_in_keyfrm_2.at(idx_2);
//        if (idx_1 == static_cast<int>(i)) {
//            matched_lms_in_keyfrm_1[idx_1] = landmarks_2.at(idx_2);
//            ++num_matches;
//        }
//    }

    return num_matches;
}

} // namespace match
} // namespace openvslam
