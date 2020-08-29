//
// Created by Patrick Liedtke on 26.08.20.
//

#ifndef OPENVSLAM_SEGMENTED_IMAGE_SEQUENCE_H
#define OPENVSLAM_SEGMENTED_IMAGE_SEQUENCE_H

#include "image_util.h"
#include <dirent.h>
#include <stdexcept>

#include <map>

class segmented_image_sequence : image_sequence {
    public:
        struct segmented_frame : image_sequence::frame {
            const std::string seg_path_;
            segmented_frame(const std::string& img_path, const std::string& seg_path, const double timestamp)
                : frame(img_path, timestamp), seg_path_(seg_path) {
            }
        };

        segmented_image_sequence(const std::string& img_dir_path, const std::string& seg_dir_path, const double fps);
        std::vector<segmented_frame> get_segmented_frames() const;

    protected:
        std::map<std::string, std::string> segmentation_files;
};

#endif //OPENVSLAM_SEGMENTED_IMAGE_SEQUENCE_H
