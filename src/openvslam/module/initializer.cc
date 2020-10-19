#include "openvslam/config.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/initialize/bearing_vector.h"
#include "openvslam/initialize/perspective.h"
#include "openvslam/match/area.h"
#include "openvslam/module/initializer.h"
#include "openvslam/optimize/global_bundle_adjuster.h"

#include <spdlog/spdlog.h>

namespace openvslam {
    namespace module {

        initializer::initializer(const camera::setup_type_t setup_type,
                                 data::map_database *map_db, data::bow_database *bow_db,
                                 const YAML::Node &yaml_node)
                : setup_type_(setup_type), map_db_(map_db), bow_db_(bow_db),
                  num_ransac_iters_(yaml_node["Initializer.num_ransac_iterations"].as<unsigned int>(100)),
                  min_num_triangulated_(yaml_node["Initializer.num_min_triangulated_pts"].as<unsigned int>(50)),
                  parallax_deg_thr_(yaml_node["Initializer.parallax_deg_threshold"].as<float>(1.0)),
                  reproj_err_thr_(yaml_node["Initializer.reprojection_error_threshold"].as<float>(4.0)),
                  num_ba_iters_(yaml_node["Initializer.num_ba_iterations"].as<unsigned int>(20)),
                  scaling_factor_(yaml_node["Initializer.scaling_factor"].as<float>(1.0)) {
            spdlog::debug("CONSTRUCT: module::initializer");
        }

        initializer::~initializer() {
            spdlog::debug("DESTRUCT: module::initializer");
        }

        void initializer::reset() {
            initializer_.reset(nullptr);
            state_ = initializer_state_t::NotReady;
            init_frm_id_ = 0;
        }

        initializer_state_t initializer::get_state() const {
            return state_;
        }

        data::keypoint_container initializer::get_initial_keypoints() const {
            return init_frm_.keypts_;
        }

        std::vector<int> initializer::get_initial_slam_matches() const {
            return init_slam_matches_;
        }

        unsigned int initializer::get_initial_frame_id() const {
            return init_frm_id_;
        }

        bool initializer::initialize(data::frame &curr_frm) {
            switch (setup_type_) {
                case camera::setup_type_t::Monocular: {
                    // construct an initializer if not constructed
                    if (state_ == initializer_state_t::NotReady) {
                        create_initializer(curr_frm);
                        return false;
                    }

                    // try to initialize
                    if (!try_initialize_for_monocular(curr_frm)) {
                        // failed
                        return false;
                    }

                    // create new map if succeeded
                    create_map_for_monocular(curr_frm);
                    break;
                }
                case camera::setup_type_t::Stereo:
                case camera::setup_type_t::RGBD: {
                    state_ = initializer_state_t::Initializing;

                    // try to initialize
                    if (!try_initialize_for_stereo(curr_frm)) {
                        // failed
                        return false;
                    }

                    // create new map if succeeded
                    create_map_for_stereo(curr_frm);
                    break;
                }
                default: {
                    throw std::runtime_error("Undefined camera setup");
                }
            }

            // check the state is succeeded or not
            if (state_ == initializer_state_t::Succeeded) {
                init_frm_id_ = curr_frm.id_;
                return true;
            } else {
                return false;
            }
        }

        void initializer::create_initializer(data::frame &curr_frm) {
            // set the initial frame
            init_frm_ = data::frame(curr_frm);

            // initialize the previously matched coordinates
            const std::vector<cv::KeyPoint> &slam_cv_points = init_frm_.undist_keypts_.get_slam_applicable_cv_keypoints();
            prev_matched_slam_applicable_coords_.resize(slam_cv_points.size());
            for (unsigned int i = 0; i < slam_cv_points.size(); ++i) {
                prev_matched_slam_applicable_coords_.at(i) = slam_cv_points.at(i).pt;
            }

            const std::vector<cv::KeyPoint> &non_slam_cv_points = init_frm_.undist_keypts_.get_slam_forbidden_cv_keypoints();
            prev_matched_slam_forbidden_coords_.resize(non_slam_cv_points.size());
            for (unsigned int i = 0; i < non_slam_cv_points.size(); ++i) {
                prev_matched_slam_forbidden_coords_.at(i) = non_slam_cv_points.at(i).pt;
            }

            // initialize matchings (init_idx -> curr_idx)
            std::fill(init_slam_matches_.begin(), init_slam_matches_.end(), -1);
            std::fill(init_non_slam_matches_.begin(), init_non_slam_matches_.end(), -1);

            // build a initializer
            initializer_.reset(nullptr);
            switch (init_frm_.camera_->model_type_) {
                case camera::model_type_t::Perspective:
                case camera::model_type_t::Fisheye: {
                    initializer_ = std::unique_ptr<initialize::perspective>(new initialize::perspective(init_frm_,
                                                                                                        num_ransac_iters_,
                                                                                                        min_num_triangulated_,
                                                                                                        parallax_deg_thr_,
                                                                                                        reproj_err_thr_));
                    break;
                }
                case camera::model_type_t::Equirectangular: {
                    initializer_ = std::unique_ptr<initialize::bearing_vector>(new initialize::bearing_vector(init_frm_,
                                                                                                              num_ransac_iters_,
                                                                                                              min_num_triangulated_,
                                                                                                              parallax_deg_thr_,
                                                                                                              reproj_err_thr_));
                    break;
                }
            }

            state_ = initializer_state_t::Initializing;
        }

        bool initializer::try_initialize_for_monocular(data::frame &curr_frm) {
            assert(state_ == initializer_state_t::Initializing);

            match::area matcher(0.9, true);
            // compute matches which are used in the initializer
            const auto num_matches = matcher.match_in_consistent_area(init_frm_, curr_frm,
                                                                      prev_matched_slam_applicable_coords_,
                                                                      init_slam_matches_, 100,
                                                                      init_frm_.undist_keypts_.get_slam_applicable_cv_keypoints(),
                                                                      curr_frm.undist_keypts_.get_slam_applicable_cv_keypoints());
            // compute additional matches which are not used for slam algorithms
            matcher.match_in_consistent_area(init_frm_, curr_frm,
                                             prev_matched_slam_forbidden_coords_,
                                             init_non_slam_matches_, 100,
                                             init_frm_.undist_keypts_.get_slam_forbidden_cv_keypoints(),
                                             curr_frm.undist_keypts_.get_slam_forbidden_cv_keypoints());

            if (num_matches < min_num_triangulated_) {
                // rebuild the initializer with the next frame
                reset();
                return false;
            }

            // try to initialize with the current frame, computes fundamental and homography matrices
            assert(initializer_);
            return initializer_->initialize(curr_frm, init_slam_matches_);
        }

        bool initializer::create_map_for_monocular(data::frame &curr_frm) {
            assert(state_ == initializer_state_t::Initializing);

            eigen_alloc_vector<Vec3_t> init_triangulated_slam_pts;
            eigen_alloc_vector<Vec3_t> init_triangulated_non_slam_pts;
            {
                assert(initializer_);
                // problem: initializer is used to compute the matrices AND to look for correct triangulation
                init_triangulated_slam_pts = initializer_->get_triangulated_pts();
                init_triangulated_non_slam_pts = triangulate_non_slam_points(curr_frm);
                const auto is_triangulated = initializer_->get_triangulated_flags();

                // init_matches has been initialized by the area matcher
                // make invalid the matchings which have not been triangulated
                for (unsigned int i = 0; i < init_slam_matches_.size(); ++i) {
                    if (init_slam_matches_.at(i) < 0) {
                        continue;
                    }
                    if (is_triangulated.at(i)) {
                        continue;
                    }
                    init_slam_matches_.at(i) = -1;
                }

                // set the camera poses
                init_frm_.set_cam_pose(Mat44_t::Identity());
                Mat44_t cam_pose_cw = Mat44_t::Identity();
                cam_pose_cw.block<3, 3>(0, 0) = initializer_->get_rotation_ref_to_cur();
                cam_pose_cw.block<3, 1>(0, 3) = initializer_->get_translation_ref_to_cur();
                curr_frm.set_cam_pose(cam_pose_cw);

                // destruct the initializer
                initializer_.reset(nullptr);
            }

            // create initial keyframes
            auto init_keyfrm = new data::keyframe(init_frm_, map_db_, bow_db_);
            auto curr_keyfrm = new data::keyframe(curr_frm, map_db_, bow_db_);

            // compute BoW representations
            init_keyfrm->compute_bow();
            curr_keyfrm->compute_bow();

            // add the keyframes to the map DB
            map_db_->add_keyframe(init_keyfrm);
            map_db_->add_keyframe(curr_keyfrm);

            // update the frame statistics
            init_frm_.ref_keyfrm_ = init_keyfrm;
            curr_frm.ref_keyfrm_ = curr_keyfrm;
            map_db_->update_frame_statistics(init_frm_, false);
            map_db_->update_frame_statistics(curr_frm, false);

            // assign 2D-3D associations of slam landmarks
            for (unsigned int init_idx = 0; init_idx < init_slam_matches_.size(); init_idx++) {
                int curr_idx = init_slam_matches_.at(init_idx);
                if (curr_idx < 0) {
                    continue;
                }

                // construct a landmark
                auto lm = new data::landmark(init_triangulated_slam_pts.at(init_idx), curr_keyfrm, map_db_);
                configure_new_landmark(curr_frm, init_keyfrm, curr_keyfrm, init_idx, curr_idx, lm);
            }

            // assign 2D-3D associations of landmarks not suitable for slam
            for (unsigned int init_idx = 0; init_idx < init_non_slam_matches_.size(); init_idx++) {
                int curr_idx = init_non_slam_matches_.at(init_idx);
                if (curr_idx < 0) {
                    continue;
                }

                // Ids are created by simply counting up. Since we already did this for slam matches, we have to add an offset to avoid duplicate ids.
                init_idx += init_slam_matches_.size();
                curr_idx += init_slam_matches_.size();

                // construct a landmark
                auto lm = new data::landmark(init_triangulated_slam_pts.at(init_idx), curr_keyfrm, map_db_);
                configure_new_landmark(curr_frm, init_keyfrm, curr_keyfrm, init_idx, curr_idx, lm);
            }

            // global bundle adjustment
            const auto global_bundle_adjuster = optimize::global_bundle_adjuster(map_db_, num_ba_iters_, true);
            global_bundle_adjuster.optimize();

            // scale the map so that the median of depths is 1.0
            const auto median_depth = init_keyfrm->compute_median_depth(
                    init_keyfrm->camera_->model_type_ == camera::model_type_t::Equirectangular);
            const auto inv_median_depth = 1.0 / median_depth;
            if (curr_keyfrm->get_num_tracked_landmarks(1) < min_num_triangulated_ && median_depth < 0) {
                spdlog::info("seems to be wrong initialization, resetting");
                state_ = initializer_state_t::Wrong;
                return false;
            }
            scale_map(init_keyfrm, curr_keyfrm, inv_median_depth * scaling_factor_);

            // update the current frame pose
            curr_frm.set_cam_pose(curr_keyfrm->get_cam_pose());

            // set the origin keyframe
            map_db_->origin_keyfrm_ = init_keyfrm;

            spdlog::info("new map created with {} landmarks: frame {} - frame {}", map_db_->get_num_landmarks(),
                         init_frm_.id_, curr_frm.id_);
            state_ = initializer_state_t::Succeeded;
            return true;
        }

        // Todo pali: association via indices has to be removed, or are these global ids?
        void initializer::configure_new_landmark(data::frame &curr_frm, data::keyframe *init_keyfrm,
                                                 data::keyframe *curr_keyfrm, unsigned int init_idx, const int curr_idx,
                                                 data::landmark *lm) const {// set the assocications to the new keyframes
            init_keyfrm->add_landmark(lm, init_idx);
            curr_keyfrm->add_landmark(lm, curr_idx);
            lm->add_observation(init_keyfrm, init_idx);
            lm->add_observation(curr_keyfrm, curr_idx);

            // update the descriptor
            lm->compute_descriptor();
            // update the geometry
            lm->update_normal_and_depth();

            // set the 2D-3D assocications to the current frame
            curr_frm.landmarks_.at(curr_idx) = lm;
            curr_frm.outlier_flags_.at(curr_idx) = false;

            // add the landmark to the map DB
            map_db_->add_landmark(lm);
        }

        void initializer::scale_map(data::keyframe *init_keyfrm, data::keyframe *curr_keyfrm, const double scale) {
            // scaling keyframes
            Mat44_t cam_pose_cw = curr_keyfrm->get_cam_pose();
            cam_pose_cw.block<3, 1>(0, 3) *= scale;
            curr_keyfrm->set_cam_pose(cam_pose_cw);

            // scaling landmarks
            const auto landmarks = init_keyfrm->get_landmarks();
            for (auto lm : landmarks) {
                if (!lm) {
                    continue;
                }
                lm->set_pos_in_world(lm->get_pos_in_world() * scale);
            }
        }

        bool initializer::try_initialize_for_stereo(data::frame &curr_frm) {
            assert(state_ == initializer_state_t::Initializing);
            // count the number of valid depths
            unsigned int num_valid_depths = std::count_if(curr_frm.depths_.begin(), curr_frm.depths_.end(),
                                                          [](const float depth) {
                                                              return 0 < depth;
                                                          });
            return min_num_triangulated_ <= num_valid_depths;
        }

        bool initializer::create_map_for_stereo(data::frame &curr_frm) {
            assert(state_ == initializer_state_t::Initializing);

            // create an initial keyframe
            curr_frm.set_cam_pose(Mat44_t::Identity());
            auto curr_keyfrm = new data::keyframe(curr_frm, map_db_, bow_db_);

            // compute BoW representation
            curr_keyfrm->compute_bow();

            // add to the map DB
            map_db_->add_keyframe(curr_keyfrm);

            // update the frame statistics
            curr_frm.ref_keyfrm_ = curr_keyfrm;
            map_db_->update_frame_statistics(curr_frm, false);

            for (unsigned int idx = 0; idx < curr_frm.num_keypts_; ++idx) {
                // add a new landmark if tht corresponding depth is valid
                const auto z = curr_frm.depths_.at(idx);
                if (z <= 0) {
                    continue;
                }

                // build a landmark
                const Vec3_t pos_w = curr_frm.triangulate_stereo(idx);
                auto lm = new data::landmark(pos_w, curr_keyfrm, map_db_);

                // set the associations to the new keyframe
                lm->add_observation(curr_keyfrm, idx);
                curr_keyfrm->add_landmark(lm, idx);

                // update the descriptor
                lm->compute_descriptor();
                // update the geometry
                lm->update_normal_and_depth();

                // set the 2D-3D associations to the current frame
                curr_frm.landmarks_.at(idx) = lm;
                curr_frm.outlier_flags_.at(idx) = false;

                // add the landmark to the map DB
                map_db_->add_landmark(lm);
            }

            // set the origin keyframe
            map_db_->origin_keyfrm_ = curr_keyfrm;

            spdlog::info("new map created with {} points: frame {}", map_db_->get_num_landmarks(), curr_frm.id_);
            state_ = initializer_state_t::Succeeded;
            return true;
        }

        eigen_alloc_vector<Vec3_t> initializer::triangulate_non_slam_points(data::frame &frame) {
            std::vector<std::pair<int, int>> ref_cur_matches;
            for (unsigned int ref_idx = 0; ref_idx < init_non_slam_matches_.size(); ++ref_idx) {
                const auto cur_idx = init_non_slam_matches_.at(ref_idx);
                if (0 <= cur_idx) {
                    ref_cur_matches.emplace_back(std::make_pair(ref_idx, cur_idx));
                }
            }

            eigen_alloc_vector<Vec3_t> matches;
            initializer_->apply_transformation_to_points(frame.undist_keypts_.get_slam_applicable_keypoints(),
                                                         init_frm_.undist_keypts_.get_slam_applicable_keypoints(),
                                                         ref_cur_matches, matches);
            return matches;
        }

    } // namespace module
} // namespace openvslam
