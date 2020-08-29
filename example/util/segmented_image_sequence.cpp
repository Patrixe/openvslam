//
// Created by Patrick Liedtke on 26.08.20.
//

#include "segmented_image_sequence.h"

segmented_image_sequence::segmented_image_sequence(const std::string& img_dir_path, const std::string& seg_dir_path, const double fps)
    : image_sequence(img_dir_path, fps) {
    DIR* dir;
    if ((dir = opendir(seg_dir_path.c_str())) == nullptr) {
        throw std::runtime_error("directory " + img_dir_path + " does not exist");
    }
    dirent* dp;

    for (dp = readdir(dir); dp != nullptr; dp = readdir(dir)) {
        const std::string seg_file_name = dp->d_name;
        if (seg_file_name.rfind("seg_", 0) != 0) {
            continue;
        }
        segmentation_files[seg_file_name.substr(4)] = seg_dir_path + "/" + seg_file_name;
    }
    closedir(dir);
}

std::vector<segmented_image_sequence::segmented_frame> segmented_image_sequence::get_segmented_frames() const {
    std::vector<segmented_frame> frames;
    for (unsigned int i = 0; i < img_file_paths_.size(); ++i) {
        try {
            std::string img_file_name = img_file_paths_.at(i).substr(img_file_paths_.at(i).rfind('/') + 1);
            frames.emplace_back(segmented_frame{img_file_paths_.at(i), segmentation_files.at(img_file_name) , (1.0 / fps_) * i});
        } catch (std::exception& e) {
            std::string img_file_name2 = img_file_paths_.at(i);
        }
    }
    return frames;
}
