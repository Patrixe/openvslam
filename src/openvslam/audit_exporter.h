//
// Created by Patrick Liedtke on 12.11.20.
//

#ifndef OPENVSLAM_AUDIT_EXPORTER_H
#define OPENVSLAM_AUDIT_EXPORTER_H

#include "openvslam/data/keyframe.h"
#include "config.h"

namespace openvslam {
        /**
         * Class used to audit the keypoints and landmarks found when a keyframe is generated.
         */
        class audit_exporter {
        public:
            audit_exporter(std::shared_ptr<config> cfg);

            void log_keyframe(const data::keyframe* keyframe);

        private:
            std::string save_path;
            std::shared_ptr<config> cfg;

            void log_keypoints(const data::keyframe *keyframe) const;

            void log_landmarks(const data::keyframe *pKeyframe);
        };
}


#endif //OPENVSLAM_AUDIT_EXPORTER_H
