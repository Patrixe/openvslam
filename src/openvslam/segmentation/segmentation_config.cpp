//
// Created by Patrick Liedtke on 27.08.20.
//

#include "segmentation_config.h"
#include <map>

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
        | (1 << 8) // vegetation
        | (1 << 9) // terrain
        | (1 << 10) // sky
        | (1 << 11) // human
        | (1 << 12) // rider
        | (1 << 13) // car
        | (1 << 14) // truck
        | (1 << 15) // bus
        | (1 << 16) // train
        | (1 << 17) // motorcycle
        | (1 << 18) // bicycle
;

const float* openvslam::segmentation_config::get_class_color(int seg_class) {
    static const std::map<int, color> class_colors{
            std::pair<int, color>(0, color{{55, 55, 55}}), // no class
            std::pair<int, color>(0, color{{255, 255, 255}}), // roads
            std::pair<int, color>(1, color{{100, 100, 100}}), // sidewalk
            std::pair<int, color>(2, color{{000, 0, 255}}), //building
            std::pair<int, color>(3, color{{255, 255, 000}}), // wall
            std::pair<int, color>(4, color{{128, 0, 128}}), // fence
            std::pair<int, color>(5, color{{128, 0, 128}}), // pole
            std::pair<int, color>(6, color{{200, 200, 000}}), // traffic light
            std::pair<int, color>(7, color{{200, 200, 000}}), // traffic sign
            std::pair<int, color>(8, color{{000, 255, 000}}), // vegetation
            std::pair<int, color>(9, color{{000, 000, 000}}), // terrain
            std::pair<int, color>(10, color{{00, 00, 00}}), // sky
            std::pair<int, color>(11, color{{255, 000, 000}}), // human
            std::pair<int, color>(12, color{{255, 000, 000}}), // rider
            std::pair<int, color>(13, color{{0, 191, 255}}), // car
            std::pair<int, color>(14, color{{000, 000, 000}}), // truck
            std::pair<int, color>(15, color{{000, 000, 000}}), // bus
            std::pair<int, color>(16, color{{000, 000, 000}}), // train
            std::pair<int, color>(17, color{{255, 000, 000}}), // motorcycle
            std::pair<int, color>(18, color{{255, 000, 000}}), // bicycle
    };

    return class_colors.at(seg_class).rgb;
}

bool openvslam::segmentation_config::allowed_for_landmark(int seg_cls) {
    return ((1 << seg_cls) & accepted_for_keypoints) || seg_cls == -1;
}

int openvslam::segmentation_config::get_segmentation_assignment_mode() {
    return 0;
}
