#include "openvslam/data/frame.h"
#include "openvslam/data/keypoint.h"
#include "openvslam/data/landmark.h"
#include "openvslam/optimize/pose_optimizer.h"
#include "openvslam/optimize/g2o/se3/pose_opt_edge_wrapper.h"
#include "openvslam/util/converter.h"

#include <vector>

#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <spdlog/spdlog.h>

namespace openvslam {
namespace optimize {

pose_optimizer::pose_optimizer(const unsigned int num_trials, const unsigned int num_each_iter)
    : num_trials_(num_trials), num_each_iter_(num_each_iter) {}

unsigned int pose_optimizer::optimize(data::frame& frm) const {
    spdlog::debug("Running pose optimizer");
    // 1. optimizerを構築

    auto linear_solver = ::g2o::make_unique<::g2o::LinearSolverEigen<::g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto block_solver = ::g2o::make_unique<::g2o::BlockSolver_6_3>(std::move(linear_solver));
    auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    ::g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    unsigned int num_init_obs = 0;

    // 2. frameをg2oのvertexに変換してoptimizerにセットする

    auto frm_vtx = new g2o::se3::shot_vertex();
    frm_vtx->setId(frm.id_);
    frm_vtx->setEstimate(util::converter::to_g2o_SE3(frm.cam_pose_cw_));
    frm_vtx->setFixed(false);
    optimizer.addVertex(frm_vtx);

    const unsigned int num_keypts = frm.undist_keypts_.size();

    // 3. landmarkのvertexをreprojection edgeで接続する

    // reprojection edgeのcontainer
    using pose_opt_edge_wrapper = g2o::se3::pose_opt_edge_wrapper<data::frame>;
    std::vector<pose_opt_edge_wrapper> pose_opt_edge_wraps;
    pose_opt_edge_wraps.reserve(num_keypts);

    // 有意水準5%のカイ2乗値
    // 自由度n=2
    constexpr float chi_sq_2D = 5.99146;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    // 自由度n=3
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

    for (auto &lm : frm.landmarks_) {
        if (lm.second->will_be_erased() || !lm.second->is_applicable_for_slam()) {
            continue;
        }

        ++num_init_obs;
        lm.second->set_outlier(false);

        // frameのvertexをreprojection edgeで接続する
        const auto& undist_keypt = frm.undist_keypts_.at(lm.first);
        const float x_right = undist_keypt.get_stereo_x_offset();
        const float inv_sigma_sq = frm.inv_level_sigma_sq_.at(undist_keypt.get_cv_keypoint().octave);
        const auto sqrt_chi_sq = (frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
                                     ? sqrt_chi_sq_2D
                                     : sqrt_chi_sq_3D;
        auto pose_opt_edge_wrap = pose_opt_edge_wrapper(&frm, frm_vtx, lm.second->get_pos_in_world(),
                                                        lm.first, undist_keypt.get_cv_keypoint().pt.x, undist_keypt.get_cv_keypoint().pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq);
        pose_opt_edge_wraps.push_back(pose_opt_edge_wrap);
        optimizer.addEdge(pose_opt_edge_wrap.edge_);
    }

    if (num_init_obs < 5) {
        spdlog::debug("PoseOptimizer: Failed with too few initial observations: {} < 5", num_init_obs);
        return 0;
    }

    // 4. robust BAを実行する

    unsigned int num_bad_obs = 0;
    for (unsigned int trial = 0; trial < num_trials_; ++trial) {
        optimizer.initializeOptimization();
        optimizer.optimize(num_each_iter_);

        num_bad_obs = 0;

        for (auto& pose_opt_edge_wrap : pose_opt_edge_wraps) {
            auto edge = pose_opt_edge_wrap.edge_;

            edge->computeError();
            frm.landmarks_.at(pose_opt_edge_wrap.idx_)->chi_squared_pose_error = edge->chi2();
            frm.landmarks_.at(pose_opt_edge_wrap.idx_)->pose_error = *edge->errorData();
//            if (frm.landmarks_.at(pose_opt_edge_wrap.idx_)->is_outlier()) {
//                edge->computeError();
//            }

            if (pose_opt_edge_wrap.is_monocular_) {
                if (chi_sq_2D < edge->chi2()) {
                    frm.landmarks_.at(pose_opt_edge_wrap.idx_)->set_outlier(true);
                    pose_opt_edge_wrap.set_as_outlier();
                    ++num_bad_obs;
                }
                else {
                    frm.landmarks_.at(pose_opt_edge_wrap.idx_)->set_outlier(false);
                    pose_opt_edge_wrap.set_as_inlier();
                }
            }
            else {
                if (chi_sq_3D < edge->chi2()) {
                    frm.landmarks_.at(pose_opt_edge_wrap.idx_)->set_outlier(true);
                    pose_opt_edge_wrap.set_as_outlier();
                    ++num_bad_obs;
                }
                else {
                    frm.landmarks_.at(pose_opt_edge_wrap.idx_)->set_outlier(false);
                    pose_opt_edge_wrap.set_as_inlier();
                }
            }

            if (trial == num_trials_ - 2) {
                edge->setRobustKernel(nullptr);
            }
        }

        if (num_init_obs - num_bad_obs < 5) {
            break;
        }
    }

    // 5. 情報を更新

    frm.set_cam_pose(frm_vtx->estimate());

    return num_init_obs - num_bad_obs;
}

} // namespace optimize
} // namespace openvslam
