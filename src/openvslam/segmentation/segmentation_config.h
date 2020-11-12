//
// Created by Patrick Liedtke on 27.08.20.
//

#ifndef OPENVSLAM_SEGMENTATION_CONFIG_H
#define OPENVSLAM_SEGMENTATION_CONFIG_H
namespace openvslam {
    struct color {
        float rgb[3];
    };

    class segmentation_config {
    public:
        static bool allowed_for_landmark(int seg_cls);

        static const float* get_class_color(int seg_class);
    };
}

#endif //OPENVSLAM_SEGMENTATION_CONFIG_H
