#include "openvslam/config.h"
#include "openvslam/system.h"
#include "openvslam/tracking_module.h"
#include "openvslam/mapping_module.h"
#include "openvslam/global_optimization_module.h"
#include "openvslam/camera/base.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/data/bow_database.h"
#include "openvslam/feature/orb_extractor.h"
#include "openvslam/match/projection.h"
#include "openvslam/module/local_map_updater.h"
#include "openvslam/util/image_converter.h"

#include <chrono>
#include <unordered_map>

#include <spdlog/spdlog.h>
#include <openvslam/feature/segmented_orb_extractor.h>

namespace openvslam {

    tracking_module::tracking_module(const std::shared_ptr<config> &cfg, system *system, data::map_database *map_db,
                                     data::bow_vocabulary *bow_vocab, data::bow_database *bow_db)
            : cfg_(cfg), camera_(cfg->camera_), system_(system), map_db_(map_db), bow_vocab_(bow_vocab),
              bow_db_(bow_db),
              initializer_(cfg->camera_->setup_type_, map_db, bow_db, cfg->yaml_node_),
              frame_tracker_(camera_, 10), relocalizer_(bow_db_), pose_optimizer_(),
              keyfrm_inserter_(cfg_->camera_->setup_type_, cfg_->true_depth_thr_, map_db, bow_db, 0,
                               cfg_->camera_->fps_) {
        spdlog::debug("CONSTRUCT: tracking_module");

        extractor_left_ = new feature::orb_extractor(cfg_->orb_params_);
        if (camera_->setup_type_ == camera::setup_type_t::Monocular) {
            ini_extractor_left_ = new feature::orb_extractor(cfg_->orb_params_);
            ini_extractor_left_->set_max_num_keypoints(ini_extractor_left_->get_max_num_keypoints() * 2);
        }
        if (camera_->setup_type_ == camera::setup_type_t::Stereo) {
            extractor_right_ = new feature::orb_extractor(cfg_->orb_params_);
        }
    }

    tracking_module::tracking_module(const std::shared_ptr<config> &cfg, system *system, data::map_database *map_db,
                                     data::bow_vocabulary *bow_vocab, data::bow_database *bow_db,
                                     const std::shared_ptr<segmentation_config> &seg_cfg, audit_exporter* auditer)
            : cfg_(cfg), camera_(cfg->camera_), system_(system), map_db_(map_db), bow_vocab_(bow_vocab),
              bow_db_(bow_db),
              initializer_(cfg->camera_->setup_type_, map_db, bow_db, cfg->yaml_node_),
              frame_tracker_(camera_, 10), relocalizer_(bow_db_), pose_optimizer_(),
              keyfrm_inserter_(cfg_->camera_->setup_type_, cfg_->true_depth_thr_, map_db, bow_db, 0,
                               cfg_->camera_->fps_), seg_cfg(seg_cfg) {
        spdlog::debug("CONSTRUCT: tracking_module");

        if (seg_cfg) {
            extractor_left_ = new feature::segmented_orb_extractor(cfg->orb_params_, seg_cfg);
            if (camera_->setup_type_ == camera::setup_type_t::Monocular) {
                ini_extractor_left_ = new feature::segmented_orb_extractor(cfg_->orb_params_, seg_cfg);
                ini_extractor_left_->set_max_num_keypoints(ini_extractor_left_->get_max_num_keypoints() * 2);
            }
            if (camera_->setup_type_ == camera::setup_type_t::Stereo) {
                extractor_right_ = new feature::segmented_orb_extractor(cfg_->orb_params_, seg_cfg);
            }

            this->auditer = auditer;
        } else {
            extractor_left_ = new feature::orb_extractor(cfg_->orb_params_);
            if (camera_->setup_type_ == camera::setup_type_t::Monocular) {
                ini_extractor_left_ = new feature::orb_extractor(cfg_->orb_params_);
                ini_extractor_left_->set_max_num_keypoints(ini_extractor_left_->get_max_num_keypoints() * 2);
            }
            if (camera_->setup_type_ == camera::setup_type_t::Stereo) {
                extractor_right_ = new feature::orb_extractor(cfg_->orb_params_);
            }
        }
    }

    tracking_module::~tracking_module() {
        delete extractor_left_;
        extractor_left_ = nullptr;
        delete extractor_right_;
        extractor_right_ = nullptr;
        delete ini_extractor_left_;
        ini_extractor_left_ = nullptr;

        spdlog::debug("DESTRUCT: tracking_module");
    }

    void tracking_module::set_mapping_module(mapping_module *mapper) {
        mapper_ = mapper;
        keyfrm_inserter_.set_mapping_module(mapper);
    }

    void tracking_module::set_global_optimization_module(global_optimization_module *global_optimizer) {
        global_optimizer_ = global_optimizer;
    }

    void tracking_module::set_mapping_module_status(const bool mapping_is_enabled) {
        std::lock_guard<std::mutex> lock(mtx_mapping_);
        mapping_is_enabled_ = mapping_is_enabled;
    }

    bool tracking_module::get_mapping_module_status() const {
        std::lock_guard<std::mutex> lock(mtx_mapping_);
        return mapping_is_enabled_;
    }

    std::map<int, data::keypoint> tracking_module::get_initial_keypoints() const {
        return initializer_.get_initial_keypoints();
    }

    const std::map<int, std::pair<data::keypoint, data::keypoint>> tracking_module::get_initial_matches() const {
        return initializer_.get_initial_slam_matches();
    }

    Mat44_t tracking_module::track_monocular_image(const cv::Mat &img, const double timestamp, const cv::Mat &mask) {
        const auto start = std::chrono::system_clock::now();

        // color conversion
        img_gray_ = img;
        util::convert_to_grayscale(img_gray_, camera_->color_order_);

        // create current frame object
        if (tracking_state_ == tracker_state_t::NotInitialized || tracking_state_ == tracker_state_t::Initializing) {
            curr_frm_ = data::frame(img_gray_, timestamp, ini_extractor_left_, bow_vocab_, camera_,
                                    cfg_->true_depth_thr_, mask);
        } else {
            curr_frm_ = data::frame(img_gray_, timestamp, extractor_left_, bow_vocab_, camera_, cfg_->true_depth_thr_,
                                    mask);
        }
        track();

        const auto end = std::chrono::system_clock::now();
        elapsed_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        return curr_frm_.cam_pose_cw_;
    }

    Mat44_t tracking_module::track_stereo_image(const cv::Mat &left_img_rect, const cv::Mat &right_img_rect,
                                                const double timestamp, const cv::Mat &mask) {
        const auto start = std::chrono::system_clock::now();

        // color conversion
        img_gray_ = left_img_rect;
        cv::Mat right_img_gray = right_img_rect;
        util::convert_to_grayscale(img_gray_, camera_->color_order_);
        util::convert_to_grayscale(right_img_gray, camera_->color_order_);

        // create current frame object
        curr_frm_ = data::frame(img_gray_, right_img_gray, timestamp, extractor_left_, extractor_right_, bow_vocab_,
                                camera_, cfg_->true_depth_thr_, mask);

        track();

        const auto end = std::chrono::system_clock::now();
        elapsed_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        return curr_frm_.cam_pose_cw_;
    }

    Mat44_t tracking_module::track_RGBD_image(const cv::Mat &img, const cv::Mat &depthmap, const double timestamp,
                                              const cv::Mat &mask) {
        const auto start = std::chrono::system_clock::now();

        // color and depth scale conversion
        img_gray_ = img;
        cv::Mat img_depth = depthmap;
        util::convert_to_grayscale(img_gray_, camera_->color_order_);
        util::convert_to_true_depth(img_depth, cfg_->depthmap_factor_);

        // create current frame object
        curr_frm_ = data::frame(img_gray_, img_depth, timestamp, extractor_left_, bow_vocab_, camera_,
                                cfg_->true_depth_thr_, mask);

        track();

        const auto end = std::chrono::system_clock::now();
        elapsed_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        return curr_frm_.cam_pose_cw_;
    }

    Mat44_t tracking_module::track_monocular_image_with_segmentation(const cv::Mat &img, const cv::Mat &seg_img,
                                                                     const double timestamp, const cv::Mat &mask) {
        const auto start = std::chrono::system_clock::now();

        // color conversion
        img_gray_ = img;
        util::convert_to_grayscale(img_gray_, camera_->color_order_);

        // create current frame object
        if (tracking_state_ == tracker_state_t::NotInitialized || tracking_state_ == tracker_state_t::Initializing) {
            curr_frm_ = data::segmentation_frame(img_gray_, seg_img, timestamp, ini_extractor_left_,
                                                 bow_vocab_, camera_, cfg_->true_depth_thr_, mask);
        } else {
            curr_frm_ = data::segmentation_frame(img_gray_, seg_img, timestamp, extractor_left_,
                                                 bow_vocab_, camera_, cfg_->true_depth_thr_, mask);
        }
        spdlog::debug("{} points in frame", curr_frm_.undist_keypts_.size());
        track();

        const auto end = std::chrono::system_clock::now();
        elapsed_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        return curr_frm_.cam_pose_cw_;
    }

    Mat44_t tracking_module::track_stereo_image_with_segmentation(const cv::Mat &left_img_rect,
                                                                  const cv::Mat &right_img_rect,
                                                                  const cv::Mat &left_seg_img,
                                                                  const cv::Mat &right_seg_img,
                                                                  const double timestamp, const cv::Mat &mask) {
        const auto start = std::chrono::system_clock::now();

        // color conversion
        img_gray_ = left_img_rect;
        cv::Mat right_img_gray = right_img_rect;
        util::convert_to_grayscale(img_gray_, camera_->color_order_);
        util::convert_to_grayscale(right_img_gray, camera_->color_order_);

        // create current frame object
        curr_frm_ = data::segmentation_frame(img_gray_, right_img_gray, left_seg_img, right_seg_img, timestamp,
                                             extractor_left_, extractor_right_, bow_vocab_, camera_,
                                             cfg_->true_depth_thr_, mask);

        track();

        const auto end = std::chrono::system_clock::now();
        elapsed_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        return curr_frm_.cam_pose_cw_;
    }

    void tracking_module::reset() {
        spdlog::info("resetting system");

        initializer_.reset();
        keyfrm_inserter_.reset();

        mapper_->request_reset();
        global_optimizer_->request_reset();

        bow_db_->clear();
        map_db_->clear();

        data::frame::next_id_ = 0;
        data::keyframe::next_id_ = 0;
        data::landmark::next_id_ = 0;

        last_reloc_frm_id_ = 0;

        tracking_state_ = tracker_state_t::NotInitialized;
    }

    void tracking_module::track() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (tracking_state_ == tracker_state_t::NotInitialized) {
            tracking_state_ = tracker_state_t::Initializing;
        }

        last_tracking_state_ = tracking_state_;

        // check if pause is requested
        check_and_execute_pause();
        while (is_paused()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }

        // LOCK the map database
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        if (tracking_state_ == tracker_state_t::Initializing) {
            if (!initialize()) {
                if (auditer) {
                    auditer->log_frame(this->curr_frm_, tracking_state_to_int(this->tracking_state_));
                }
                return;
            }

            // update the reference keyframe, local keyframes, and local landmarks
            update_local_map();

            // pass all of the keyframes to the mapping module
            const auto keyfrms = map_db_->get_all_keyframes();
            for (const auto keyfrm : keyfrms) {
                mapper_->queue_keyframe(keyfrm);
            }

            // state transition to Tracking mode
            tracking_state_ = tracker_state_t::Tracking;
            if (auditer) {
                auditer->log_frame(this->curr_frm_, tracking_state_to_int(this->tracking_state_));
            }
        } else {
            // apply replace of landmarks observed in the last frame
            apply_landmark_replace();
            // update the camera pose of the last frame
            // because the mapping module might optimize the camera pose of the last frame's reference keyframe
            update_last_frame();

            // set the reference keyframe of the current frame
            curr_frm_.ref_keyfrm_ = ref_keyfrm_;

            auto succeeded = track_current_frame();
            if (auditer) {
                auditer->log_frame(this->curr_frm_, tracking_state_to_int(this->tracking_state_));
            }

            // update the local map and optimize the camera pose of the current frame
            if (succeeded) {
                update_local_map();
                succeeded = optimize_current_frame_with_local_map();
            }

            // update the motion model
            if (succeeded) {
                update_motion_model();
            }

            // state transition
            tracking_state_ = succeeded ? tracker_state_t::Tracking : tracker_state_t::Lost;

            // update the frame statistics
            map_db_->update_frame_statistics(curr_frm_, tracking_state_ == tracker_state_t::Lost);

            // if tracking is failed within 5.0 sec after initialization, reset the system
            constexpr float init_retry_thr = 5.0;
            if (tracking_state_ == tracker_state_t::Lost
                && curr_frm_.id_ - initializer_.get_initial_frame_id() < camera_->fps_ * init_retry_thr) {
                spdlog::info("tracking lost within {} sec after initialization", init_retry_thr);
                system_->request_reset();

                if (auditer) {
                    auditer->log_frame(this->curr_frm_, tracking_state_to_int(this->tracking_state_));
                }
                return;
            }

            // show message if tracking has been lost
            if (last_tracking_state_ != tracker_state_t::Lost && tracking_state_ == tracker_state_t::Lost) {
                spdlog::info("tracking lost: frame {}", curr_frm_.id_);
            }

            // check to insert the new keyframe derived from the current frame
            if (succeeded && new_keyframe_is_needed()) {
                insert_new_keyframe();
            }

            // tidy up observations
            for (auto lm = curr_frm_.landmarks_.begin(); lm != curr_frm_.landmarks_.end();) {
                if (lm->second->is_outlier()) {
                    curr_frm_.landmarks_.erase(lm++);
                } else {
                    ++lm;
                }
            }
        }

        // store the relative pose from the reference keyframe to the current frame
        // to update the camera pose at the beginning of the next tracking process
        if (curr_frm_.cam_pose_cw_is_valid_) {
            last_cam_pose_from_ref_keyfrm_ = curr_frm_.cam_pose_cw_ * curr_frm_.ref_keyfrm_->get_cam_pose_inv();
        }

        // update last frame
        last_frm_ = curr_frm_;
    }

    bool tracking_module::initialize() {
        // try to initialize with the current frame
        initializer_.initialize(curr_frm_);

        // if map building was failed -> reset the map database
        if (initializer_.get_state() == module::initializer_state_t::Wrong) {
            // reset
            system_->request_reset();
            return false;
        }

        // if initializing was failed -> try to initialize with the next frame
        if (initializer_.get_state() != module::initializer_state_t::Succeeded) {
            return false;
        }

        // succeeded
        return true;
    }

    bool tracking_module::track_current_frame() {
        bool succeeded = false;
        if (tracking_state_ == tracker_state_t::Tracking) {
            spdlog::debug("--Before tracking: {} landmarks in ref-keyframe", ref_keyfrm_->get_landmarks().size());
            // Tracking mode
            if (velocity_is_valid_ && last_reloc_frm_id_ + 2 < curr_frm_.id_) {
                // if the motion model is valid
                succeeded = frame_tracker_.motion_based_track(curr_frm_, last_frm_, velocity_);
            }
            if (!succeeded) {
                succeeded = frame_tracker_.bow_match_based_track(curr_frm_, last_frm_, ref_keyfrm_);
            }
            if (!succeeded) {
                succeeded = frame_tracker_.robust_match_based_track(curr_frm_, last_frm_, ref_keyfrm_);
            }
        } else {
            // Lost mode
            // try to relocalize
            succeeded = relocalizer_.relocalize(curr_frm_);
            if (succeeded) {
                last_reloc_frm_id_ = curr_frm_.id_;
            }
        }
        return succeeded;
    }

    void tracking_module::update_motion_model() {
        if (last_frm_.cam_pose_cw_is_valid_) {
            Mat44_t last_frm_cam_pose_wc = Mat44_t::Identity();
            last_frm_cam_pose_wc.block<3, 3>(0, 0) = last_frm_.get_rotation_inv();
            last_frm_cam_pose_wc.block<3, 1>(0, 3) = last_frm_.get_cam_center();
            velocity_is_valid_ = true;
            velocity_ = curr_frm_.cam_pose_cw_ * last_frm_cam_pose_wc;
        } else {
            velocity_is_valid_ = false;
            velocity_ = Mat44_t::Identity();
        }
    }

    void tracking_module::apply_landmark_replace() {
        for (auto lm : last_frm_.landmarks_) {
            auto replaced_lm = lm.second->get_replaced();
            if (replaced_lm) {
                last_frm_.landmarks_[lm.first] = replaced_lm;
            }
        }
    }

    void tracking_module::update_last_frame() {
        auto last_ref_keyfrm = last_frm_.ref_keyfrm_;
        if (!last_ref_keyfrm) {
            return;
        }
        last_frm_.set_cam_pose(last_cam_pose_from_ref_keyfrm_ * last_ref_keyfrm->get_cam_pose());
    }

    bool tracking_module::optimize_current_frame_with_local_map() {
        // acquire more 2D-3D matches by reprojecting the local landmarks to the current frame
        search_local_landmarks();

        // optimize the pose
        pose_optimizer_.optimize(curr_frm_);

        // count up the number of tracked landmarks
        num_tracked_lms_ = 0;
        for (auto iter = curr_frm_.landmarks_.begin(); iter != curr_frm_.landmarks_.end();) {
            auto &lm = *iter;
            if (!lm.second->is_outlier()) {
                // the observation has been considered as inlier in the pose optimization
                assert(lm.second->has_observation());
                // count up
                ++num_tracked_lms_;
                // increment the number of tracked frame
                lm.second->increase_num_observed();
            } else {
                // the observation has been considered as outlier in the pose optimization
                // remove the observation
                curr_frm_.landmarks_.erase(iter++);
                continue;
            }
            ++iter;
        }

        constexpr unsigned int num_tracked_lms_thr = 20;

        // if recently relocalized, use the more strict threshold
        if (curr_frm_.id_ < last_reloc_frm_id_ + camera_->fps_ && num_tracked_lms_ < 2 * num_tracked_lms_thr) {
            spdlog::debug("local map tracking failed: {} matches < {}", num_tracked_lms_, 2 * num_tracked_lms_thr);
            return false;
        }

        // check the threshold of the number of tracked landmarks
        if (num_tracked_lms_ < num_tracked_lms_thr) {
            spdlog::debug("local map tracking failed: {} matches < {}", num_tracked_lms_, num_tracked_lms_thr);
            return false;
        }

        return true;
    }

    void tracking_module::update_local_map() {
        spdlog::debug("update local map: {} landmarks in ref-keyframe", curr_frm_.landmarks_.size());
        // clean landmark associations
        for (auto lm = curr_frm_.landmarks_.begin(); lm != curr_frm_.landmarks_.end();) {
            if (lm->second->will_be_erased()) {
                curr_frm_.landmarks_.erase(lm++);
                continue;
            } else {
                ++lm;
            }
        }
        spdlog::debug("update local map: {} landmarks in ref-keyframe after erasing", curr_frm_.landmarks_.size());

        // acquire the current local map
        // This is not a global limit for the map size, but regulates how many frames (up to two levels) are connected to the current frame.
        constexpr unsigned int max_num_local_keyfrms = 60;
        auto local_map_updater = module::local_map_updater(curr_frm_, max_num_local_keyfrms);
        if (!local_map_updater.acquire_local_map()) {
            return;
        }
        // update the variables
        local_keyfrms_ = local_map_updater.get_local_keyframes();
        local_landmarks_ = local_map_updater.get_local_landmarks();
        auto nearest_covisibility = local_map_updater.get_nearest_covisibility();

        // update the reference keyframe for the current frame
        if (nearest_covisibility) {
            ref_keyfrm_ = nearest_covisibility;
            curr_frm_.ref_keyfrm_ = ref_keyfrm_;
        }

        map_db_->set_local_landmarks(local_landmarks_);
    }

    void tracking_module::search_local_landmarks() {
        // select the landmarks which can be reprojected from the ones observed in the current frame
        for (auto lm : curr_frm_.landmarks_) {
            if (lm.second->will_be_erased()) {
                continue;
            }

            // this landmark cannot be reprojected
            // because already observed in the current frame
            lm.second->is_observable_in_tracking_ = false;
            lm.second->identifier_in_local_lm_search_ = curr_frm_.id_;

            // this landmark is observable from the current frame
            lm.second->increase_num_observable();
        }

        bool found_proj_candidate = false;
        // temporary variables
        Vec2_t reproj;
        float x_right;
        unsigned int pred_scale_level;
        for (auto lm : local_landmarks_) {
            // avoid the landmarks which cannot be reprojected (== observed in the current frame)
            if (lm->identifier_in_local_lm_search_ == curr_frm_.id_) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            // check the observability
            if (curr_frm_.can_observe(lm, 0.5, reproj, x_right, pred_scale_level)) {
                // pass the temporary variables
                lm->reproj_in_tracking_ = reproj;
                lm->x_right_in_tracking_ = x_right;
                lm->scale_level_in_tracking_ = pred_scale_level;

                // this landmark can be reprojected
                lm->is_observable_in_tracking_ = true;

                // this landmark is observable from the current frame
                lm->increase_num_observable();

                found_proj_candidate = true;
            } else {
                // this landmark cannot be reprojected
                lm->is_observable_in_tracking_ = false;
            }
        }

        if (!found_proj_candidate) {
            return;
        }

        // acquire more 2D-3D matches by projecting the local landmarks to the current frame
        match::projection projection_matcher(0.8);
        const float margin = (curr_frm_.id_ < last_reloc_frm_id_ + 2)
                             ? 20.0
                             : ((camera_->setup_type_ == camera::setup_type_t::RGBD)
                                ? 10.0
                                : 5.0);
        projection_matcher.match_frame_and_landmarks(curr_frm_, local_landmarks_, margin);
    }

    bool tracking_module::new_keyframe_is_needed() const {
        if (!mapping_is_enabled_) {
            return false;
        }

        // cannnot insert the new keyframe in a second after relocalization
        const auto num_keyfrms = map_db_->get_num_keyframes();
        if (cfg_->camera_->fps_ < num_keyfrms && curr_frm_.id_ < last_reloc_frm_id_ + cfg_->camera_->fps_) {
            return false;
        }

        // check the new keyframe is needed
        return keyfrm_inserter_.new_keyframe_is_needed(curr_frm_, num_tracked_lms_, *ref_keyfrm_);
    }

    void tracking_module::insert_new_keyframe() {
        spdlog::debug("Inserting keyframe with {} landmarks", curr_frm_.landmarks_.size());
        // insert the new keyframe
        const auto ref_keyfrm = keyfrm_inserter_.insert_new_keyframe(curr_frm_);
        // set the reference keyframe with the new keyframe
        ref_keyfrm_ = ref_keyfrm ? ref_keyfrm : ref_keyfrm_;
        curr_frm_.ref_keyfrm_ = ref_keyfrm_;
    }

    void tracking_module::request_pause() {
        std::lock_guard<std::mutex> lock1(mtx_pause_);
        pause_is_requested_ = true;
    }

    bool tracking_module::pause_is_requested() const {
        std::lock_guard<std::mutex> lock(mtx_pause_);
        return pause_is_requested_;
    }

    bool tracking_module::is_paused() const {
        std::lock_guard<std::mutex> lock(mtx_pause_);
        return is_paused_;
    }

    void tracking_module::resume() {
        std::lock_guard<std::mutex> lock(mtx_pause_);

        is_paused_ = false;
        pause_is_requested_ = false;

        spdlog::info("resume tracking module");
    }

    bool tracking_module::check_and_execute_pause() {
        std::lock_guard<std::mutex> lock(mtx_pause_);
        if (pause_is_requested_) {
            is_paused_ = true;
            spdlog::info("pause tracking module");
            return true;
        } else {
            return false;
        }
    }

    bool tracking_module::is_tracking() {
        return tracking_state_ == tracker_state_t::Tracking;
    }

    int tracking_module::tracking_state_to_int(openvslam::tracker_state_t& state) {
        switch (state) {
            case tracker_state_t::NotInitialized:
                return 0;
            case tracker_state_t::Initializing:
                return 1;
            case tracker_state_t::Tracking:
                return 2;
            case tracker_state_t::Lost:
                return 3;
        }
    }
} // namespace openvslam
