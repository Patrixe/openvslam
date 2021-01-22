//
// Created by Patrick Liedtke on 12.11.20.
//

#include "audit_exporter.h"

#include "openvslam/data/landmark.h"

#include <fstream>
#include <spdlog/spdlog.h>

namespace openvslam {
    audit_exporter::audit_exporter(std::shared_ptr<config> cfg) : cfg(std::move(cfg)), save_path(cfg->audit_save_path){
    }

    void audit_exporter::log_keyframe(const data::keyframe *keyframe) {
        if (cfg->do_audit) {
            log_keypoints(keyframe);
            log_landmarks(keyframe);
        }
    }

    void audit_exporter::log_keypoints(const data::keyframe *keyframe) const {
        std::ofstream point_file;
        try {
            point_file.open(save_path + "/undist_keypoints_" + std::to_string(keyframe->id_) + ".audit", std::ios::trunc);
            for (auto &keypoint : keyframe->undist_keypts_) {
                point_file << keypoint.first << "," << keypoint.second.get_segmentation_class() <<
                           "," << keypoint.second.get_cv_keypoint().pt.x <<
                           "," << keypoint.second.get_cv_keypoint().pt.y <<
                           "," << keypoint.second.get_cv_keypoint().size <<
                           "," << keypoint.second.get_cv_keypoint().octave <<
                           "," << keypoint.second.get_cv_keypoint().response <<
                           "," << keypoint.second.get_cv_keypoint().angle;
                point_file << "\n";
            }
        } catch (const std::fstream::failure &f) {
            spdlog::error("Failed to write to keypoint file: {}", f.what());
        }

        point_file.close();
    }

    void audit_exporter::log_landmarks(const data::keyframe *keyframe) {
        std::ofstream landmark_file;
        try {
            landmark_file.open(save_path + "/landmarks" + std::to_string(keyframe->id_) + ".audit", std::ios::trunc);
            landmark_file << "meta," << keyframe->timestamp_ << "," << keyframe->src_frm_id_ << "\n";
            for (auto &landmark : keyframe->get_landmarks()) {
                const auto &keypoint = keyframe->undist_keypts_.at(landmark.first);
                landmark_file << landmark.second->id_ << "," << landmark.second->get_segmentation_class() <<
                              "," << keypoint.get_cv_keypoint().pt.x <<
                              "," << keypoint.get_cv_keypoint().pt.y <<
                              "," << keypoint.get_cv_keypoint().size <<
                              "," << keypoint.get_cv_keypoint().octave <<
                              "," << keypoint.get_cv_keypoint().response <<
                              "," << keypoint.get_cv_keypoint().angle <<
                              "," << landmark.second->pose_error <<
                              "," << landmark.second->chi_squared_pose_error <<
                              "," << landmark.second->ba_error <<
                              "," << landmark.second->chi_squared_ba_error;
                landmark_file << "\n";
            }
        } catch (const std::fstream::failure &f) {
            spdlog::error("Failed to write to landmark file: {}", f.what());
        }

        landmark_file.close();
    }
}
