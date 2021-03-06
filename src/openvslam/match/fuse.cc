#include "openvslam/match/fuse.h"
#include "openvslam/camera/base.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/keypoint.h"
#include "openvslam/data/landmark.h"

#include <vector>
#include <unordered_set>

namespace openvslam {
namespace match {

unsigned int fuse::detect_duplication(data::keyframe* keyfrm, const Mat44_t& Sim3_cw, const std::map<int, data::landmark*>& landmarks_to_check,
                                      const float margin, std::map<data::landmark*, data::landmark*>& duplicated_lms_in_keyfrm) {
    unsigned int num_fused = 0;

    // Sim3を分解してSE3にする
    const Mat33_t s_rot_cw = Sim3_cw.block<3, 3>(0, 0);
    const auto s_cw = std::sqrt(s_rot_cw.block<1, 3>(0, 0).dot(s_rot_cw.block<1, 3>(0, 0)));
    const Mat33_t rot_cw = s_rot_cw / s_cw;
    const Vec3_t trans_cw = Sim3_cw.block<3, 1>(0, 3) / s_cw;
    const Vec3_t cam_center = -rot_cw.transpose() * trans_cw;

    const auto valid_lms_in_keyfrm = keyfrm->get_valid_landmarks();

    for (const auto &landmark_to_check : landmarks_to_check) {
        auto* lm = landmark_to_check.second;
        if (lm->will_be_erased()) {
            continue;
        }
        // この3次元点とkeyframeの特徴点がすでに対応している場合は，再投影・統合する必要がないのでスルー
        if (valid_lms_in_keyfrm.count(lm)) {
            continue;
        }

        // グローバル基準の3次元点座標
        const Vec3_t pos_w = lm->get_pos_in_world();

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
        const auto max_cam_to_lm_dist = lm->get_max_valid_distance();
        const auto min_cam_to_lm_dist = lm->get_min_valid_distance();

        if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist) {
            continue;
        }

        // 3次元点の平均観測ベクトルとの角度を計算し，閾値(60deg)より大きければ破棄
        const Vec3_t obs_mean_normal = lm->get_obs_mean_normal();

        if (cam_to_lm_vec.dot(obs_mean_normal) < 0.5 * cam_to_lm_dist) {
            continue;
        }

        // 3次元点を再投影した点が存在するcellの特徴点を取得
        const int pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, keyfrm);

        const auto neighbouring_keypoints = keyfrm->get_keypoints_in_cell(reproj(0), reproj(1), margin * keyfrm->scale_factors_.at(pred_scale_level));

        if (neighbouring_keypoints.empty()) {
            continue;
        }

        // descriptorが最も近い特徴点を探す
        const auto lm_desc = lm->get_descriptor();

        unsigned int best_dist = MAX_HAMMING_DIST;
        std::reference_wrapper<const data::keypoint> best_matching_keypoint = neighbouring_keypoints[0];

        for (const auto neighbouring_keypoint_ref : neighbouring_keypoints) {
            const auto scale_level = neighbouring_keypoint_ref.get().get_cv_keypoint().octave;

            // TODO: keyfrm->get_keypts_in_cell()でスケールの判断をする
            if (scale_level < pred_scale_level - 1 || pred_scale_level < scale_level) {
                continue;
            }

            const auto& desc = neighbouring_keypoint_ref.get().get_orb_descriptor_as_cv_mat();

            const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

            if (hamm_dist < best_dist) {
                best_dist = hamm_dist;
                best_matching_keypoint = neighbouring_keypoint_ref;
            }
        }

        if (HAMMING_DIST_THR_LOW < best_dist) {
            continue;
        }

        const std::map<int, data::landmark *> &keyframe_landmarks = keyfrm->get_landmarks();
        if (keyframe_landmarks.find(best_matching_keypoint.get().get_id()) != keyframe_landmarks.end()) {
            auto* lm_in_keyfrm = keyfrm->get_landmark(best_matching_keypoint.get().get_id());
            // keyframeのbest_idxに対応する3次元点が存在する -> 重複している場合
            if (!lm_in_keyfrm->will_be_erased()) {
                duplicated_lms_in_keyfrm[landmark_to_check.second] = lm_in_keyfrm;
            }
        }
        else {
            // keyframeのbest_idxに対応する3次元点が存在しない
            // 観測情報を追加
            lm->add_observation(keyfrm, best_matching_keypoint.get().get_id());
            keyfrm->add_landmark(lm, best_matching_keypoint.get().get_id());
        }

        ++num_fused;
    }

    return num_fused;
}

template<typename T>
unsigned int fuse::replace_duplication(data::keyframe* keyfrm, const T& landmarks_to_check, const float margin) {
    unsigned int num_fused = 0;

    const Mat33_t rot_cw = keyfrm->get_rotation();
    const Vec3_t trans_cw = keyfrm->get_translation();
    const Vec3_t cam_center = keyfrm->get_cam_center();

    for (const auto &lm : landmarks_to_check) {
        if (lm->will_be_erased()) {
            continue;
        }
        if (lm->is_observed_in_keyframe(keyfrm)) {
            continue;
        }

        // グローバル基準の3次元点座標
        const Vec3_t pos_w = lm->get_pos_in_world();

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
        const auto max_cam_to_lm_dist = lm->get_max_valid_distance();
        const auto min_cam_to_lm_dist = lm->get_min_valid_distance();

        if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist) {
            continue;
        }

        // 3次元点の平均観測ベクトルとの角度を計算し，閾値(60deg)より大きければ破棄
        const Vec3_t obs_mean_normal = lm->get_obs_mean_normal();

        if (cam_to_lm_vec.dot(obs_mean_normal) < 0.5 * cam_to_lm_dist) {
            continue;
        }

        // 3次元点を再投影した点が存在するcellの特徴点を取得
        const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, keyfrm);

        const auto neighbouring_keypoints = keyfrm->get_keypoints_in_cell(reproj(0), reproj(1), margin * keyfrm->scale_factors_.at(pred_scale_level));

        if (neighbouring_keypoints.empty()) {
            continue;
        }

        // descriptorが最も近い特徴点を探す
        const auto lm_desc = lm->get_descriptor();

        unsigned int best_dist = MAX_HAMMING_DIST;
        std::reference_wrapper<const data::keypoint> best_matching_keypoint = neighbouring_keypoints[0];

        for (const auto neighbour_keypoint_ref : neighbouring_keypoints) {
            const auto& keypt = neighbour_keypoint_ref.get();

            const auto scale_level = static_cast<unsigned int>(keypt.get_cv_keypoint().octave);

            // TODO: keyfrm->get_keypts_in_cell()でスケールの判断をする
            if (scale_level < pred_scale_level - 1 || pred_scale_level < scale_level) {
                continue;
            }

            if (keypt.get_stereo_x_offset() > 0) {
                // stereo matchが存在する場合は自由度3の再投影誤差を計算する
                const auto e_x = reproj(0) - keypt.get_cv_keypoint().pt.x;
                const auto e_y = reproj(1) - keypt.get_cv_keypoint().pt.y;
                const auto e_x_right = x_right - keypt.get_stereo_x_offset();
                const auto reproj_error_sq = e_x * e_x + e_y * e_y + e_x_right * e_x_right;

                // 自由度n=3
                constexpr float chi_sq_3D = 7.81473;
                if (chi_sq_3D < reproj_error_sq * keyfrm->inv_level_sigma_sq_.at(scale_level)) {
                    continue;
                }
            }
            else {
                // stereo matchが存在しない場合は自由度2の再投影誤差を計算する
                const auto e_x = reproj(0) - keypt.get_cv_keypoint().pt.x;
                const auto e_y = reproj(1) - keypt.get_cv_keypoint().pt.y;
                const auto reproj_error_sq = e_x * e_x + e_y * e_y;

                // 自由度n=2
                constexpr float chi_sq_2D = 5.99146;
                if (chi_sq_2D < reproj_error_sq * keyfrm->inv_level_sigma_sq_.at(scale_level)) {
                    continue;
                }
            }

            const auto& desc = keypt.get_orb_descriptor_as_cv_mat();

            const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

            if (hamm_dist < best_dist) {
                best_dist = hamm_dist;
                best_matching_keypoint = keypt;
            }
        }

        if (HAMMING_DIST_THR_LOW < best_dist) {
            continue;
        }

        const std::map<int, data::landmark *> &keyframe_landmarks = keyfrm->get_landmarks();
        if (keyframe_landmarks.find(best_matching_keypoint.get().get_id()) != keyframe_landmarks.end()) {
            auto* lm_in_keyfrm = keyfrm->get_landmark(best_matching_keypoint.get().get_id());
            // keyframeのbest_idxに対応する3次元点が存在する -> 重複している場合
            if (!lm_in_keyfrm->will_be_erased()) {
                // より信頼できる(=観測数が多い)3次元点で置き換える
                if (lm->num_observations() < lm_in_keyfrm->num_observations()) {
                    // lm_in_keyfrmで置き換える
                    lm->replace(lm_in_keyfrm);
                }
                else {
                    // lmで置き換える
                    lm_in_keyfrm->replace(lm);
                }
            }
        }
        else {
            // keyframeのbest_idxに対応する3次元点が存在しない
            // 観測情報を追加
            lm->add_observation(keyfrm, best_matching_keypoint.get().get_id());
            keyfrm->add_landmark(lm, best_matching_keypoint.get().get_id());
        }

        ++num_fused;
    }

    return num_fused;
}

unsigned int fuse::replace_duplication(data::keyframe* keyfrm, const std::map<int, data::landmark *> &landmarks_to_check, const float margin) {
    unsigned int num_fused = 0;

    const Mat33_t rot_cw = keyfrm->get_rotation();
    const Vec3_t trans_cw = keyfrm->get_translation();
    const Vec3_t cam_center = keyfrm->get_cam_center();

    for (const auto &lm : landmarks_to_check) {
        if (lm.second->will_be_erased()) {
            continue;
        }
        if (lm.second->is_observed_in_keyframe(keyfrm)) {
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

        const auto neighbouring_keypoints = keyfrm->get_keypoints_in_cell(reproj(0), reproj(1), margin * keyfrm->scale_factors_.at(pred_scale_level));

        if (neighbouring_keypoints.empty()) {
            continue;
        }

        // descriptorが最も近い特徴点を探す
        const auto &lm_desc = lm.second->get_descriptor();

        unsigned int best_dist = MAX_HAMMING_DIST;
        std::reference_wrapper<const data::keypoint> best_matching_keypoint = neighbouring_keypoints[0];

        for (const auto &neighbour_keypoint_ref : neighbouring_keypoints) {
            const auto& keypt = neighbour_keypoint_ref.get();

            const auto scale_level = static_cast<unsigned int>(keypt.get_cv_keypoint().octave);

            // TODO: keyfrm->get_keypts_in_cell()でスケールの判断をする
            if (scale_level < pred_scale_level - 1 || pred_scale_level < scale_level) {
                continue;
            }

            if (keypt.get_stereo_x_offset() > 0) {
                // stereo matchが存在する場合は自由度3の再投影誤差を計算する
                const auto e_x = reproj(0) - keypt.get_cv_keypoint().pt.x;
                const auto e_y = reproj(1) - keypt.get_cv_keypoint().pt.y;
                const auto e_x_right = x_right - keypt.get_stereo_x_offset();
                const auto reproj_error_sq = e_x * e_x + e_y * e_y + e_x_right * e_x_right;

                // 自由度n=3
                constexpr float chi_sq_3D = 7.81473;
                if (chi_sq_3D < reproj_error_sq * keyfrm->inv_level_sigma_sq_.at(scale_level)) {
                    continue;
                }
            }
            else {
                // stereo matchが存在しない場合は自由度2の再投影誤差を計算する
                const auto e_x = reproj(0) - keypt.get_cv_keypoint().pt.x;
                const auto e_y = reproj(1) - keypt.get_cv_keypoint().pt.y;
                const auto reproj_error_sq = e_x * e_x + e_y * e_y;

                // 自由度n=2
                constexpr float chi_sq_2D = 5.99146;
                if (chi_sq_2D < reproj_error_sq * keyfrm->inv_level_sigma_sq_.at(scale_level)) {
                    continue;
                }
            }

            const auto& desc = keypt.get_orb_descriptor_as_cv_mat();

            const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

            if (hamm_dist < best_dist) {
                best_dist = hamm_dist;
                best_matching_keypoint = keypt;
            }
        }

        if (HAMMING_DIST_THR_LOW < best_dist) {
            continue;
        }

        const std::map<int, data::landmark *> &keyframe_landmarks = keyfrm->get_landmarks();
        if (keyframe_landmarks.find(best_matching_keypoint.get().get_id()) != keyframe_landmarks.end()) {
            auto* lm_in_keyfrm = keyfrm->get_landmark(best_matching_keypoint.get().get_id());
            // keyframeのbest_idxに対応する3次元点が存在する -> 重複している場合
            if (!lm_in_keyfrm->will_be_erased()) {
                // より信頼できる(=観測数が多い)3次元点で置き換える
                if (lm.second->num_observations() < lm_in_keyfrm->num_observations()) {
                    // lm_in_keyfrmで置き換える
                    lm.second->replace(lm_in_keyfrm);
                }
                else {
                    // lmで置き換える
                    lm_in_keyfrm->replace(lm.second);
                }
            }
        }
        else {
            // keyframeのbest_idxに対応する3次元点が存在しない
            // 観測情報を追加
            lm.second->add_observation(keyfrm, best_matching_keypoint.get().get_id());
            keyfrm->add_landmark(lm.second, best_matching_keypoint.get().get_id());
        }

        ++num_fused;
    }

    return num_fused;
}

// 明示的に実体化しておく
template unsigned int fuse::replace_duplication(data::keyframe*, const std::vector<data::landmark*>&, const float);
template unsigned int fuse::replace_duplication(data::keyframe*, const std::unordered_set<data::landmark*>&, const float);

} // namespace match
} // namespace openvslam
