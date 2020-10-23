#include "openvslam/data/frame.h"
#include "openvslam/initialize/bearing_vector.h"
#include "openvslam/solve/essential_solver.h"

#include <spdlog/spdlog.h>

namespace openvslam {
namespace initialize {

bearing_vector::bearing_vector(const data::frame& ref_frm,
                               const unsigned int num_ransac_iters, const unsigned int min_num_triangulated,
                               const float parallax_deg_thr, const float reproj_err_thr)
    : base(ref_frm, num_ransac_iters, min_num_triangulated, parallax_deg_thr, reproj_err_thr) {
    spdlog::debug("CONSTRUCT: initialize::bearing_vector");
}

bearing_vector::~bearing_vector() {
    spdlog::debug("DESTRUCT: initialize::bearing_vector");
}

bool bearing_vector::initialize(const data::frame& cur_frm, const std::map<int, std::pair<data::keypoint, data::keypoint>> &ref_matches_with_cur) {
    // set the current camera model
    cur_camera_ = cur_frm.camera_;
    // store the keypoints and bearings
    cur_undist_keypts_.reserve(ref_matches_with_cur.size());
    cur_bearings_.reserve(ref_matches_with_cur.size());

    ref_undist_keypts_.reserve(ref_matches_with_cur.size());
    ref_bearings_.reserve(ref_matches_with_cur.size());
    ref_undist_keypt_ids.clear();
    ref_undist_keypt_ids.reserve(ref_matches_with_cur.size());

    // align matching information
    ref_cur_matches_.clear();
    ref_cur_matches_.reserve(ref_matches_with_cur.size());

    int running_index = 0;
    for (const std::pair<int, std::pair<data::keypoint, data::keypoint>> match : ref_matches_with_cur) {
        // add points to the lists of point an bearing. We dont need all points of a frame, as only matches are taken into account
        ref_undist_keypts_.emplace_back(match.second.first.get_cv_keypoint());
        ref_undist_keypt_ids.emplace_back(match.first);
        ref_bearings_.emplace_back(match.second.first.get_bearing());
        // same for current frame
        cur_undist_keypts_.emplace_back(match.second.second.get_cv_keypoint());
        cur_bearings_.emplace_back(match.second.second.get_bearing());
        // lastly register as match. Since we technically inserted the points ordered, it looks strange.
        ref_cur_matches_.emplace_back(std::make_pair(running_index, running_index));

        running_index++;
    }

    // compute an E matrix
    auto essential_solver = solve::essential_solver(ref_bearings_, cur_bearings_, ref_cur_matches_);
    essential_solver.find_via_ransac(num_ransac_iters_);

    // reconstruct map if the solution is valid
    if (essential_solver.solution_is_valid()) {
        const Mat33_t E_ref_to_cur = essential_solver.get_best_E_21();
        const auto is_inlier_match = essential_solver.get_inlier_matches();
        return reconstruct_with_E(E_ref_to_cur, is_inlier_match);
    }
    else {
        return false;
    }
}

bool bearing_vector::reconstruct_with_E(const Mat33_t& E_ref_to_cur, const std::vector<bool>& is_inlier_match) {
    // found the most plausible pose from the FOUR hypothesis computed from the E matrix

    // decompose the E matrix
    eigen_alloc_vector<Mat33_t> init_rots;
    eigen_alloc_vector<Vec3_t> init_transes;
    if (!solve::essential_solver::decompose(E_ref_to_cur, init_rots, init_transes)) {
        return false;
    }

    assert(init_rots.size() == 4);
    assert(init_transes.size() == 4);

    const auto pose_is_found = find_most_plausible_pose(init_rots, init_transes, is_inlier_match, false);
    if (!pose_is_found) {
        return false;
    }

    spdlog::info("initialization succeeded with E");
    return true;
}

} // namespace initialize
} // namespace openvslam
