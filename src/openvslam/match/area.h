#ifndef OPENVSLAM_MATCH_AREA_H
#define OPENVSLAM_MATCH_AREA_H

#include "openvslam/match/base.h"

namespace openvslam {

    namespace data {
        class frame;
    } // namespace data

    namespace match {

        class area final : public base {
        public:
            area(const float lowe_ratio, const bool check_orientation)
                    : base(lowe_ratio, check_orientation) {}

            ~area() final = default;

            unsigned int match_in_consistent_area(data::frame &frm_2, std::map<int, cv::Point2f> &prev_matched_pts,
                                                  std::map<int, std::pair<data::keypoint, data::keypoint>> &found_matches,
                                                  int margin,
                                                  const std::vector<data::keypoint> &frame_1_slam_cv_points,
                                                  const std::vector<data::keypoint> &frame_2_slam_cv_points);
        };

    } // namespace match
} // namespace openvslam

#endif // OPENVSLAM_MATCH_AREA_H
