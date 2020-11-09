#include "openvslam/data/frame.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/keypoint.h"
#include "openvslam/data/landmark.h"
#include "openvslam/match/bow_tree.h"
#include "openvslam/match/angle_checker.h"

#ifdef USE_DBOW2
#include <DBoW2/FeatureVector.h>
#include <spdlog/spdlog.h>

#else
#include <fbow/fbow.h>
#endif

namespace openvslam {
namespace match {

unsigned int bow_tree::match_frame_and_keyframe(data::keyframe* keyfrm, data::frame& frm, std::map<int, data::landmark*>& matched_lms_in_frm) const {
    unsigned int num_matches = 0;

    angle_checker<int> angle_checker;

    const auto &keyframe_landmarks = keyfrm->get_landmarks();

#ifdef USE_DBOW2
    DBoW2::FeatureVector::const_iterator keyframe_feature_itr = keyfrm->bow_feat_vec_.begin();
    DBoW2::FeatureVector::const_iterator frame_feature_itr = frm.bow_feat_vec_.begin();
    const DBoW2::FeatureVector::const_iterator kryfrm_end = keyfrm->bow_feat_vec_.end();
    const DBoW2::FeatureVector::const_iterator frm_end = frm.bow_feat_vec_.end();
#else
    fbow::BoWFeatVector::const_iterator keyframe_feature_itr = keyfrm->bow_feat_vec_.begin();
    fbow::BoWFeatVector::const_iterator frame_feature_itr = frm.bow_feat_vec_.begin();
    const fbow::BoWFeatVector::const_iterator kryfrm_end = keyfrm->bow_feat_vec_.end();
    const fbow::BoWFeatVector::const_iterator frm_end = frm.bow_feat_vec_.end();
#endif
    while (keyframe_feature_itr != kryfrm_end && frame_feature_itr != frm_end) {
        // BoW treeのノード番号(first)が一致しているか確認する
        if (keyframe_feature_itr->first == frame_feature_itr->first) {
            // BoW treeのノード番号(first)が一致していれば，
            // 実際に特徴点index(second)を持ってきて対応しているか確認する
            const auto& keyfrm_indices = keyframe_feature_itr->second;
            const auto& frm_indices = frame_feature_itr->second;

            // check if the bow vectors match
            for (const auto keyfrm_idx : keyfrm_indices) {
                // keyfrm_idxの特徴点と3次元点が対応していない場合はスルーする
                if (keyframe_landmarks.find(keyfrm->get_keypoint_id_from_bow_id(keyfrm_idx)) == keyframe_landmarks.end()) {
                    continue;
                }

                const data::keypoint &keypoint_keyframe = keyfrm->undist_keypts_.at(keyfrm->get_keypoint_id_from_bow_id(keyfrm_idx));
                if (!keypoint_keyframe.is_applicable_for_slam()) {
                    continue;
                }

                auto* keyframe_landmark = keyframe_landmarks.at(keyfrm->get_keypoint_id_from_bow_id(keyfrm_idx));
                if (keyframe_landmark->will_be_erased()) {
                    continue;
                }

                const auto& keyfrm_desc = keypoint_keyframe.get_orb_descriptor_as_cv_mat();

                unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                int best_frm_idx = -1;
                unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;

                for (const auto frm_idx : frm_indices) {
                    if (matched_lms_in_frm.find(frm.get_keypoint_id_from_bow_id(frm_idx)) != matched_lms_in_frm.end()) {
                        continue;
                    }

                    data::keypoint &keypoint_frame = frm.undist_keypts_.at(frm.get_keypoint_id_from_bow_id(frm_idx));
                    if (!keypoint_frame.is_applicable_for_slam()) {
                        continue;
                    }

                    const auto& frm_desc = keypoint_frame.get_orb_descriptor_as_cv_mat();

                    const auto hamm_dist = compute_descriptor_distance_32(keyfrm_desc, frm_desc);

                    if (hamm_dist < best_hamm_dist) {
                        second_best_hamm_dist = best_hamm_dist;
                        best_hamm_dist = hamm_dist;
                        best_frm_idx = frm_idx;
                    }
                    else if (hamm_dist < second_best_hamm_dist) {
                        second_best_hamm_dist = hamm_dist;
                    }
                }

                if (HAMMING_DIST_THR_LOW < best_hamm_dist) {
                    continue;
                }

                // ratio test
                if (lowe_ratio_ * second_best_hamm_dist < static_cast<float>(best_hamm_dist)) {
                    continue;
                }

                matched_lms_in_frm[frm.get_keypoint_id_from_bow_id(best_frm_idx)] = keyframe_landmark;

                if (check_orientation_) {
                    const auto delta_angle
                        = keyfrm->keypts_.at(keyfrm->get_keypoint_id_from_bow_id(keyfrm_idx)).get_cv_keypoint().angle
                                - frm.keypts_.at(frm.get_keypoint_id_from_bow_id(best_frm_idx)).get_cv_keypoint().angle;
                    angle_checker.append_delta_angle(delta_angle, frm.get_keypoint_id_from_bow_id(best_frm_idx));
                }

                ++num_matches;
            }

            ++keyframe_feature_itr;
            ++frame_feature_itr;
        }
        else if (keyframe_feature_itr->first < frame_feature_itr->first) {
            // keyfrm_itrのノード番号のほうが小さいので，ノード番号が合うところまでイテレータkeyfrm_itrをすすめる
            keyframe_feature_itr = keyfrm->bow_feat_vec_.lower_bound(frame_feature_itr->first);
        }
        else {
            // frm_itrのノード番号のほうが小さいので，ノード番号が合うところまでイテレータfrm_itrをすすめる
            frame_feature_itr = frm.bow_feat_vec_.lower_bound(keyframe_feature_itr->first);
        }
    }

    if (check_orientation_) {
        const auto invalid_matches = angle_checker.get_invalid_matches();
        for (const auto invalid_idx : invalid_matches) {
            matched_lms_in_frm.erase(invalid_idx);
            --num_matches;
        }
    }

    return num_matches;
}

unsigned int bow_tree::match_keyframes(data::keyframe* keyfrm_1, data::keyframe* keyfrm_2, std::map<int, data::landmark*>& matched_lms_in_keyfrm_1) const {
    unsigned int num_matches = 0;

    angle_checker<int> angle_checker;

    const auto &keyfrm_1_lms = keyfrm_1->get_landmarks();
    const auto &keyfrm_2_lms = keyfrm_2->get_landmarks();

    // keyframe2の特徴点のうち，keyfram1の特徴点と対応が取れているものはtrueにする
    // NOTE: sizeはkeyframe2の特徴点に一致
    std::vector<bool> is_already_matched_in_keyfrm_2(keyfrm_2_lms.size(), false);

#ifdef USE_DBOW2
    DBoW2::FeatureVector::const_iterator itr_1 = keyfrm_1->bow_feat_vec_.begin();
    DBoW2::FeatureVector::const_iterator itr_2 = keyfrm_2->bow_feat_vec_.begin();
    const DBoW2::FeatureVector::const_iterator itr_1_end = keyfrm_1->bow_feat_vec_.end();
    const DBoW2::FeatureVector::const_iterator itr_2_end = keyfrm_2->bow_feat_vec_.end();
#else
    fbow::BoWFeatVector::const_iterator itr_1 = keyfrm_1->bow_feat_vec_.begin();
    fbow::BoWFeatVector::const_iterator itr_2 = keyfrm_2->bow_feat_vec_.begin();
    const fbow::BoWFeatVector::const_iterator itr_1_end = keyfrm_1->bow_feat_vec_.end();
    const fbow::BoWFeatVector::const_iterator itr_2_end = keyfrm_2->bow_feat_vec_.end();
#endif

    while (itr_1 != itr_1_end && itr_2 != itr_2_end) {
        // BoW treeのノード番号(first)が一致しているか確認する
        if (itr_1->first == itr_2->first) {
            // BoW treeのノード番号(first)が一致していれば，
            // 実際に特徴点index(second)を持ってきて対応しているか確認する
            const auto& keyfrm_1_indices = itr_1->second;
            const auto& keyfrm_2_indices = itr_2->second;

            for (const auto idx_1 : keyfrm_1_indices) {
                const data::keypoint &keypoint_keyframe1 = keyfrm_1->undist_keypts_.at(keyfrm_1->get_keypoint_id_from_bow_id(idx_1));
                if (!keypoint_keyframe1.is_applicable_for_slam()) {
                    continue;
                }
                // keyfrm_1の特徴点と3次元点が対応していない場合はスルーする
                if (keyfrm_1_lms.find(keyfrm_1->get_keypoint_id_from_bow_id(idx_1)) == keyfrm_1_lms.end()) {
                    continue;
                }

                auto* lm_1 = keyfrm_1_lms.at(keyfrm_1->get_keypoint_id_from_bow_id(idx_1));
                if (lm_1->will_be_erased()) {
                    continue;
                }

                const auto& desc_1 = keypoint_keyframe1.get_orb_descriptor_as_cv_mat();

                unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                int best_idx_2 = -1;
                unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;

                for (const auto idx_2 : keyfrm_2_indices) {
                    // keyfrm_2の特徴点と3次元点が対応していない場合はスルーする
                    if (keyfrm_2_lms.find(keyfrm_2->get_keypoint_id_from_bow_id(idx_2)) == keyfrm_2_lms.end()) {
                        continue;
                    }
                    const data::keypoint &keypoint_keyframe2 = keyfrm_2->undist_keypts_.at(keyfrm_2->get_keypoint_id_from_bow_id(idx_2));
                    auto* lm_2 = keyfrm_2_lms.at(keypoint_keyframe2.get_id());
                    if (lm_2->will_be_erased()) {
                        continue;
                    }

                    if (is_already_matched_in_keyfrm_2.at(keypoint_keyframe2.get_id())) {
                        continue;
                    }

                    const auto& desc_2 = keypoint_keyframe2.get_orb_descriptor_as_cv_mat();

                    const auto hamm_dist = compute_descriptor_distance_32(desc_1, desc_2);

                    if (hamm_dist < best_hamm_dist) {
                        second_best_hamm_dist = best_hamm_dist;
                        best_hamm_dist = hamm_dist;
                        best_idx_2 = idx_2;
                    }
                    else if (hamm_dist < second_best_hamm_dist) {
                        second_best_hamm_dist = hamm_dist;
                    }
                }

                if (HAMMING_DIST_THR_LOW < best_hamm_dist) {
                    continue;
                }

                // ratio test
                if (lowe_ratio_ * second_best_hamm_dist < static_cast<float>(best_hamm_dist)) {
                    continue;
                }

                // 対応情報を記録する
                // keyframe1のidx_1とkeyframe2のbest_idx_2が対応している
                matched_lms_in_keyfrm_1[keypoint_keyframe1.get_id()] = keyfrm_2_lms.at(keyfrm_2->get_keypoint_id_from_bow_id(best_idx_2));
                // keyframe2のbest_idx_2はすでにkeyframe1の特徴点と対応している
                is_already_matched_in_keyfrm_2.at(keyfrm_2->get_keypoint_id_from_bow_id(best_idx_2)) = true;

                num_matches++;

                if (check_orientation_) {
                    const auto delta_angle
                        = keypoint_keyframe1.get_cv_keypoint().angle - keyfrm_2->keypts_.at(keyfrm_2->get_keypoint_id_from_bow_id(best_idx_2)).get_cv_keypoint().angle;
                    angle_checker.append_delta_angle(delta_angle, keypoint_keyframe1.get_id());
                }
            }

            ++itr_1;
            ++itr_2;
        }
        else if (itr_1->first < itr_2->first) {
            itr_1 = keyfrm_1->bow_feat_vec_.lower_bound(itr_2->first);
        }
        else {
            itr_2 = keyfrm_2->bow_feat_vec_.lower_bound(itr_1->first);
        }
    }

    if (check_orientation_) {
        const auto invalid_matches = angle_checker.get_invalid_matches();
        for (const auto invalid_idx : invalid_matches) {
            matched_lms_in_keyfrm_1.erase(invalid_idx);
            --num_matches;
        }
    }

    return num_matches;
}

} // namespace match
} // namespace openvslam
