//
// Created by Patrick Liedtke on 27.08.20.
//

#ifndef OPENVSLAM_SEGMENTATION_CONFIG_H
#define OPENVSLAM_SEGMENTATION_CONFIG_H
namespace openvslam {
    class segmentation_config {
    public:
        bool allowed_for_landmark(int seg_cls);
    };
}

#endif //OPENVSLAM_SEGMENTATION_CONFIG_H
