//
// Created by Patrick Liedtke on 27.08.20.
//

#include "segmentation_config.h"

static const unsigned int accepted_for_keypoints =
//        0;
        (1 << 0) // roads
        | (1 << 1) // sidewalk
        | (1 << 2) // building
        | (1 << 3) // wall
        | (1 << 4) // fence
        | (1 << 5) // pole
        | (1 << 6) // traffic light
        | (1 << 7) // traffic sign
//        | (1 << 8) // vegetation
        | (1 << 9) // terrain
        | (1 << 10) // sky
        | (1 << 11) // human
        | (1 << 12) // rider
//        | (1 << 13) // car
        | (1 << 14) // truck
        | (1 << 15) // bus
        | (1 << 16) // train
        | (1 << 17) // motorcycle
//        | (1 << 18) // bycicle
;

bool openvslam::segmentation_config::allowed_for_landmark(int seg_cls) {
    return ((1 << seg_cls) & accepted_for_keypoints);
}
