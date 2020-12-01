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

        /**
         * Indicates the method that should be used to determine a features segmentation class.
         *
         * @return integer representing the selected mode:
         *  * 0: center of feature determines the features segmentation class.
         *  * 1: Majority voting of all pixels taken into account to get the actual segmentation class
         */
        static int get_segmentation_assignment_mode();
    };
}

#endif //OPENVSLAM_SEGMENTATION_CONFIG_H
