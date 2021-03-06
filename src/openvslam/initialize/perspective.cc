#include "openvslam/camera/perspective.h"
#include "openvslam/camera/fisheye.h"
#include "openvslam/data/frame.h"
#include "openvslam/initialize/perspective.h"
#include "openvslam/solve/homography_solver.h"
#include "openvslam/solve/fundamental_solver.h"

#include <thread>

#include <spdlog/spdlog.h>

namespace openvslam {
    namespace initialize {

        perspective::perspective(const data::frame &ref_frm,
                                 const unsigned int num_ransac_iters, const unsigned int min_num_triangulated,
                                 const float parallax_deg_thr, const float reproj_err_thr)
                : base(ref_frm, num_ransac_iters, min_num_triangulated, parallax_deg_thr, reproj_err_thr),
                  ref_cam_matrix_(get_camera_matrix(ref_frm.camera_)) {
            spdlog::debug("CONSTRUCT: initialize::perspective");
        }

        perspective::~perspective() {
            spdlog::debug("DESTRUCT: initialize::perspective");
        }

/**
 *
 * @param cur_frm
 * @param ref_matches_with_cur is a list of slam applicable points, where the elements position is the index of the
 *          point in the slam applicable list of the init frame. The value at this point is the index of the point in
 *          the slam applicable list of the current frame.
 * @return
 */
        bool perspective::initialize(const data::frame &cur_frm,
                                     const std::map<int, std::pair<data::keypoint, data::keypoint>> &ref_matches_with_cur) {
            // set the current camera model
            cur_camera_ = cur_frm.camera_;
            // store the keypoints and bearings
            cur_undist_keypts_.clear();
            cur_undist_keypts_.reserve(ref_matches_with_cur.size());

            cur_bearings_.clear();

            ref_undist_keypts_.clear();
            ref_undist_keypts_.reserve(ref_matches_with_cur.size());
            ref_undist_keypt_ids.clear();
            ref_undist_keypt_ids.reserve(ref_matches_with_cur.size());
            ref_bearings_.clear();

            // align matching information
            ref_cur_matches_.clear();
            ref_cur_matches_.reserve(ref_matches_with_cur.size());

            int running_index = 0;
            for (const std::pair<int, std::pair<data::keypoint, data::keypoint>> match : ref_matches_with_cur) {
                // add points to the lists of point an bearing. We dont need all points of a frame, as only matches are taken into account
                ref_undist_keypts_.emplace_back(match.second.first.get_cv_keypoint());
                ref_undist_keypt_ids.emplace_back(match.first);
                ref_bearings_.insert(std::pair<int, Vec3_t >(running_index, match.second.first.get_bearing()));
                // same for current frame
                cur_undist_keypts_.emplace_back(match.second.second.get_cv_keypoint());
                cur_bearings_.insert(std::pair<int, Vec3_t >(running_index, match.second.second.get_bearing()));
                // lastly register as match. Since we technically inserted the points ordered, it looks strange.
                ref_cur_matches_.emplace_back(std::make_pair(running_index, running_index));

                running_index++;
            }

            // set the current camera matrix
            cur_cam_matrix_ = get_camera_matrix(cur_frm.camera_);

            // compute H and F matrices
            auto homography_solver = solve::homography_solver(ref_undist_keypts_, cur_undist_keypts_, ref_cur_matches_,
                                                              1.0);
            auto fundamental_solver = solve::fundamental_solver(ref_undist_keypts_, cur_undist_keypts_,
                                                                ref_cur_matches_, 1.0);
            std::thread thread_for_H(&solve::homography_solver::find_via_ransac, &homography_solver, num_ransac_iters_,
                                     true);
            std::thread thread_for_F(&solve::fundamental_solver::find_via_ransac, &fundamental_solver,
                                     num_ransac_iters_, true);
            thread_for_H.join();
            thread_for_F.join();

            // compute a score
            const auto score_H = homography_solver.get_best_score();
            const auto score_F = fundamental_solver.get_best_score();
            const float rel_score_H = score_H / (score_H + score_F);

            // select a case according to the score
            if (0.40 < rel_score_H && homography_solver.solution_is_valid()) {
                const Mat33_t H_ref_to_cur = homography_solver.get_best_H_21();
                const auto is_inlier_match = homography_solver.get_inlier_matches();
                return reconstruct_with_H(H_ref_to_cur, is_inlier_match);
            } else if (fundamental_solver.solution_is_valid()) {
                const Mat33_t F_ref_to_cur = fundamental_solver.get_best_F_21();
                const auto is_inlier_match = fundamental_solver.get_inlier_matches();
                return reconstruct_with_F(F_ref_to_cur, is_inlier_match);
            } else {
                return false;
            }
        }

        bool perspective::reconstruct_with_H(const Mat33_t &H_ref_to_cur, const std::vector<bool> &is_inlier_match) {
            // found the most plausible pose from the EIGHT hypothesis computed from the H matrix

            // decompose the H matrix
            eigen_alloc_vector<Mat33_t> init_rots;
            eigen_alloc_vector<Vec3_t> init_transes;
            eigen_alloc_vector<Vec3_t> init_normals;
            if (!solve::homography_solver::decompose(H_ref_to_cur, ref_cam_matrix_, cur_cam_matrix_, init_rots,
                                                     init_transes, init_normals)) {
                return false;
            }

            assert(init_rots.size() == 8);
            assert(init_transes.size() == 8);

            const auto pose_is_found = find_most_plausible_pose(init_rots, init_transes, is_inlier_match, true);
            if (!pose_is_found) {
                return false;
            }

            spdlog::info("initialization succeeded with H");
            return true;
        }

        bool perspective::reconstruct_with_F(const Mat33_t &F_ref_to_cur, const std::vector<bool> &is_inlier_match) {
            // found the most plausible pose from the FOUR hypothesis computed from the F matrix

            // decompose the F matrix
            eigen_alloc_vector<Mat33_t> init_rots;
            eigen_alloc_vector<Vec3_t> init_transes;
            if (!solve::fundamental_solver::decompose(F_ref_to_cur, ref_cam_matrix_, cur_cam_matrix_, init_rots,
                                                      init_transes)) {
                return false;
            }

            assert(init_rots.size() == 4);
            assert(init_transes.size() == 4);

            const auto pose_is_found = find_most_plausible_pose(init_rots, init_transes, is_inlier_match, true);
            if (!pose_is_found) {
                return false;
            }

            spdlog::info("initialization succeeded with F");
            return true;
        }

        Mat33_t perspective::get_camera_matrix(camera::base *camera) {
            switch (camera->model_type_) {
                case camera::model_type_t::Perspective: {
                    auto c = static_cast<camera::perspective *>(camera);
                    return c->eigen_cam_matrix_;
                }
                case camera::model_type_t::Fisheye: {
                    auto c = static_cast<camera::fisheye *>(camera);
                    return c->eigen_cam_matrix_;
                }
                default: {
                    throw std::runtime_error("Cannot get a camera matrix from the camera model");
                }
            }
        }

    } // namespace initialize
} // namespace openvslam
