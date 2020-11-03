#include "openvslam/camera/base.h"
#include "openvslam/data/frame.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/match/bow_tree.h"
#include "openvslam/match/projection.h"
#include "openvslam/match/robust.h"
#include "openvslam/module/frame_tracker.h"

#include <spdlog/spdlog.h>

namespace openvslam {
namespace module {

frame_tracker::frame_tracker(camera::base* camera, const unsigned int num_matches_thr)
    : camera_(camera), num_matches_thr_(num_matches_thr), pose_optimizer_() {}

bool frame_tracker::motion_based_track(data::frame& curr_frm, const data::frame& last_frm, const Mat44_t& velocity) const {
    spdlog::debug("Attempting motion based tracking using ProjectionMatcher");
    match::projection projection_matcher(0.9, true);

    // motion modelを使って姿勢の初期値を設定
    curr_frm.set_cam_pose(velocity * last_frm.cam_pose_cw_);

    // 2D-3D対応を初期化
//    std::fill(curr_frm.landmarks_.begin(), curr_frm.landmarks_.end(), nullptr);
    curr_frm.landmarks_.clear();

    // last frameで見えている3次元点を再投影して2D-3D対応を見つける
    const float margin = (camera_->setup_type_ != camera::setup_type_t::Stereo) ? 20 : 10;
    auto num_matches = projection_matcher.match_current_and_last_frames(curr_frm, last_frm, margin);
    spdlog::debug("ProjectionMatcher: Found {} matches", num_matches);
    if (num_matches < num_matches_thr_) {
        // marginを広げて再探索
//        std::fill(curr_frm.landmarks_.begin(), curr_frm.landmarks_.end(), nullptr);
        curr_frm.landmarks_.clear();
        num_matches = projection_matcher.match_current_and_last_frames(curr_frm, last_frm, 2 * margin);
        spdlog::debug("ProjectionMatcher: Rerunning with extended margin, found {} matches", num_matches);
    }

    if (num_matches < num_matches_thr_) {
        spdlog::debug("ProjectionMatcher: motion based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // pose optimization
    spdlog::debug("ProjectionMatcher: Pose optimizing");
    pose_optimizer_.optimize(curr_frm);

    // outlierを除く
    const auto num_valid_matches = discard_outliers(curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("ProjectionMatcher: motion based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        spdlog::debug("ProjectionMatcher: Successful");
        return true;
    }
}

bool frame_tracker::bow_match_based_track(data::frame& curr_frm, const data::frame& last_frm, data::keyframe* ref_keyfrm) const {
    spdlog::debug("Attempting bow match tracking using bow matcher");
    match::bow_tree bow_matcher(0.7, true);

    // BoW matchを行うのでBoWを計算しておく
    curr_frm.compute_bow();

    // keyframeとframeで2D対応を探して，frameの特徴点とkeyframeで観測している3次元点の対応を得る
    std::map<int, data::landmark*> matched_lms_in_curr;
    auto num_matches = bow_matcher.match_frame_and_keyframe(ref_keyfrm, curr_frm, matched_lms_in_curr);
    spdlog::debug("BowMatcher: Found {} ({}) matches", matched_lms_in_curr.size(), num_matches);
    if (num_matches < num_matches_thr_) {
        spdlog::debug("bow match based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // 2D-3D対応情報を更新
    curr_frm.landmarks_ = matched_lms_in_curr;

    // pose optimization
    // 初期値は前のフレームの姿勢
    curr_frm.set_cam_pose(last_frm.cam_pose_cw_);
    spdlog::debug("BowMatcher: Pose optimizing");
    pose_optimizer_.optimize(curr_frm);

    // outlierを除く
    const auto num_valid_matches = discard_outliers(curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("BowMatcher: bow match based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        spdlog::debug("BowMatcher: Successful");
        return true;
    }
}

bool frame_tracker::robust_match_based_track(data::frame& curr_frm, const data::frame& last_frm, data::keyframe* ref_keyfrm) const {
    spdlog::debug("Running robust match tracking using robust matcher");
    match::robust robust_matcher(0.8, false);

    // keyframeとframeで2D対応を探して，frameの特徴点とkeyframeで観測している3次元点の対応を得る
    std::map<int, data::landmark*> matched_lms_in_curr;
    auto num_matches = robust_matcher.match_frame_and_keyframe(curr_frm, ref_keyfrm, matched_lms_in_curr);

    if (num_matches < num_matches_thr_) {
        spdlog::debug("robust match based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // 2D-3D対応情報を更新
    curr_frm.landmarks_ = matched_lms_in_curr;

    // pose optimization
    // 初期値は前のフレームの姿勢
    curr_frm.set_cam_pose(last_frm.cam_pose_cw_);
    pose_optimizer_.optimize(curr_frm);

    // outlierを除く
    const auto num_valid_matches = discard_outliers(curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("robust match based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        spdlog::debug("RobustMatcher: Successful");
        return true;
    }
}

unsigned int frame_tracker::discard_outliers(data::frame& curr_frm) const {
    unsigned int num_valid_matches = 0;

    for (auto iter = curr_frm.landmarks_.begin(); iter != curr_frm.landmarks_.end();) {
        if (iter->second->is_outlier()) {
            iter->second->is_observable_in_tracking_ = false;
            iter->second->identifier_in_local_lm_search_ = curr_frm.id_;
            curr_frm.landmarks_.erase(iter++);
            continue;
        }

        ++num_valid_matches;
        iter++;
    }

    return num_valid_matches;
}

} // namespace module
} // namespace openvslam
