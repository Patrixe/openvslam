#ifndef OPENVSLAM_PUBLISH_FRAME_PUBLISHER_H
#define OPENVSLAM_PUBLISH_FRAME_PUBLISHER_H

#include "openvslam/config.h"
#include "openvslam/tracking_module.h"
#include "openvslam/data/landmark.h"

#include <mutex>
#include <vector>

#include <opencv2/core/core.hpp>

namespace openvslam {

class tracking_module;

namespace data {
class map_database;
} // namespace data

namespace publish {

class frame_publisher {
public:
    /**
     * Constructor
     */
    frame_publisher(const std::shared_ptr<config>& cfg, data::map_database* map_db,
                    unsigned int img_width = 1024);

    /**
     * Destructor
     */
    virtual ~frame_publisher();

    /**
     * Update tracking information
     * NOTE: should be accessed from system thread
     */
    void update(tracking_module* tracker);

    /**
     * Get the current image with tracking information
     * NOTE: should be accessed from viewer thread
     */
    cv::Mat draw_frame(bool draw_text = true);

protected:
    unsigned int draw_initial_points(cv::Mat& img,
                                     const std::map<int, std::pair<data::keypoint, data::keypoint>> &init_matches,
                                     float mag = 1.0) const;

    unsigned int draw_tracked_points(cv::Mat& img, const data::keypoint_container &curr_keypts,
                                     const std::map<int, bool>& is_tracked, const bool mapping_is_enabled,
                                     const float mag = 1.0) const;

    void draw_info_text(cv::Mat& img, tracker_state_t tracking_state, unsigned int num_tracked,
                        double elapsed_ms, bool mapping_is_enabled) const;

    // colors (BGR)
    const cv::Scalar mapping_color_{0, 255, 255};
    const cv::Scalar localization_color_{255, 255, 0};

    //! config
    std::shared_ptr<config> cfg_;
    //! map database
    data::map_database* map_db_;
    //! maximum size of output images
    const int img_width_;

    // -------------------------------------------
    //! mutex to access variables below
    std::mutex mtx_;

    //! raw img
    cv::Mat img_;
    //! tracking state
    tracker_state_t tracking_state_;

    //! initial keypoints
    std::map<int, data::keypoint> init_keypts_;
    //! matching between initial frame and current frame
    std::map<int, std::pair<data::keypoint, data::keypoint>> init_matches_;

    //! current keypoints
    data::keypoint_container curr_keypts_;

    //! elapsed time for tracking
    double elapsed_ms_ = 0.0;

    //! mapping module status
    bool mapping_is_enabled_;

    //! tracking flag for each current keypoint
    std::map<int, bool> is_tracked_;
};

} // namespace publish
} // namespace openvslam

#endif // OPENVSLAM_PUBLISH_FRAME_PUBLISHER_H
