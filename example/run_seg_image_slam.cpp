//
// Created by Patrick Liedtke on 26.08.20.
//
#include "util/image_util.h"

#ifdef USE_PANGOLIN_VIEWER

#include "pangolin_viewer/viewer.h"

#elif USE_SOCKET_PUBLISHER
#include "socket_publisher/publisher.h"
#endif

#include "openvslam/system.h"
#include "openvslam/config.h"
#include "util/segmented_image_sequence.h"

#include <iostream>
#include <chrono>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>
#include <openvslam/segmentation_system.h>

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

void mono_tracking(const std::shared_ptr<openvslam::config> &cfg,
                   const std::string &vocab_file_path, const std::string &image_dir_path,
                   const std::string &seg_dir_path,
                   const std::string &mask_img_path,
                   const unsigned int frame_skip, const bool no_sleep, const bool auto_term,
                   const bool eval_log, const std::string &map_db_path) {
    // load the mask image
    const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

    const segmented_image_sequence sequence(image_dir_path, seg_dir_path, cfg->camera_->fps_);
    const auto frames = sequence.get_segmented_frames();

    // build a SLAM system
    std::shared_ptr<openvslam::segmentation_config> seg_cfg = std::make_shared<openvslam::segmentation_config>();
    openvslam::segmentation_system SLAM(cfg, vocab_file_path, seg_cfg);
    // startup the SLAM process
    SLAM.startup();

#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

    std::vector<double> track_times;
    track_times.reserve(frames.size());

    // run the SLAM in another thread
    std::thread thread([&]() {
        for (unsigned int i = 0; i < frames.size(); ++i) {
            const auto &frame = frames.at(i);
            const auto img = cv::imread(frame.img_path_, cv::IMREAD_UNCHANGED);
            const auto seg_img = cv::imread(frame.seg_path_, cv::IMREAD_UNCHANGED);

            if (seg_img.empty()) {
                std::cout << "No segmentation information for frame " << frame.img_path_ << std::endl;
            }

            const auto tp_1 = std::chrono::steady_clock::now();

            if (!img.empty() && (i % frame_skip == 0)) {
                // input the current frame and estimate the camera pose
                SLAM.feed_monocular_frame(img, seg_img, frame.timestamp_, mask);
            }

            const auto tp_2 = std::chrono::steady_clock::now();

            const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
            if (i % frame_skip == 0) {
                track_times.push_back(track_time);
            }

            // wait until the timestamp of the next frame
            if (!no_sleep && i < frames.size() - 1) {
                const auto wait_time = frames.at(i + 1).timestamp_ - (frame.timestamp_ + track_time);
                if (0.0 < wait_time) {
                    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<unsigned int>(wait_time * 1e6)));
                }
            }

            // check if the termination of SLAM system is requested or not
            if (SLAM.terminate_is_requested()) {
                break;
            }
        }

        // wait until the loop BA is finished
        while (SLAM.loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }

        // automatically close the viewer
#ifdef USE_PANGOLIN_VIEWER
        if (auto_term) {
            viewer.request_terminate();
        }
#elif USE_SOCKET_PUBLISHER
        if (auto_term) {
            publisher.request_terminate();
        }
#endif
    });

    // run the viewer in the current thread
#ifdef USE_PANGOLIN_VIEWER
    viewer.run();
#elif USE_SOCKET_PUBLISHER
    publisher.run();
#endif

    thread.join();

    // shutdown the SLAM process
    SLAM.shutdown();

    if (eval_log) {
        // output the trajectories for evaluation
        SLAM.save_frame_trajectory("frame_trajectory.txt", "TUM");
        SLAM.save_keyframe_trajectory("keyframe_trajectory.txt", "TUM");
        // output the tracking times for evaluation
        std::ofstream ofs("track_times.txt", std::ios::out);
        if (ofs.is_open()) {
            for (const auto track_time : track_times) {
                ofs << track_time << std::endl;
            }
            ofs.close();
        }
    }

    if (!map_db_path.empty()) {
        // output the map database
        SLAM.save_map_database(map_db_path);
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
}

void stereo_tracking(const std::shared_ptr<openvslam::config> &cfg,
                     const std::string &vocab_file_path, const std::string &image_dir_path_left,
                     const std::string &image_dir_path_right, const std::string &seg_dir_path_left,
                     const std::string &seg_dir_path_right,
                     const std::string &mask_img_path,
                     const unsigned int frame_skip, const bool no_sleep, const bool auto_term,
                     const bool eval_log, const std::string &map_db_path) {
    const segmented_image_sequence sequence_left(image_dir_path_left, seg_dir_path_left, cfg->camera_->fps_);
    const segmented_image_sequence sequence_right(image_dir_path_right, seg_dir_path_right, cfg->camera_->fps_);
    const auto frames_left = sequence_left.get_segmented_frames();
    const auto frames_right = sequence_right.get_segmented_frames();

    // build a SLAM system
    std::shared_ptr<openvslam::segmentation_config> seg_cfg = std::make_shared<openvslam::segmentation_config>();
    openvslam::segmentation_system SLAM(cfg, vocab_file_path, seg_cfg);
    // startup the SLAM process
    SLAM.startup();

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

    std::vector<double> track_times;
    track_times.reserve(frames_left.size());

    // run the SLAM in another thread
    std::thread thread([&]() {
        for (unsigned int i = 0; i < frames_left.size(); ++i) {
            const auto &left_frame = frames_left.at(i);
            const auto &right_frame = frames_right.at(i);
            const auto left_img = cv::imread(left_frame.img_path_, cv::IMREAD_UNCHANGED);
            const auto right_img = cv::imread(right_frame.img_path_, cv::IMREAD_UNCHANGED);
            const auto left_seg = cv::imread(left_frame.seg_path_, cv::IMREAD_UNCHANGED);
            const auto right_seg = cv::imread(right_frame.seg_path_, cv::IMREAD_UNCHANGED);

            const auto tp_1 = std::chrono::steady_clock::now();

            if (!left_img.empty() && !right_img.empty() && (i % frame_skip == 0)) {
                // input the current frame and estimate the camera pose
                SLAM.feed_stereo_frame(left_img, right_img, left_seg, right_seg, left_frame.timestamp_, cv::Mat{});
            }

            const auto tp_2 = std::chrono::steady_clock::now();

            const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
            if (i % frame_skip == 0) {
                track_times.push_back(track_time);
            }

            // wait until the timestamp of the next frame
            if (!no_sleep && i < frames_left.size() - 1) {
                const auto wait_time = frames_left.at(i + 1).timestamp_ - (left_frame.timestamp_ + track_time);
                if (0.0 < wait_time) {
                    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<unsigned int>(wait_time * 1e6)));
                }
            }

            // check if the termination of SLAM system is requested or not
            if (SLAM.terminate_is_requested()) {
                break;
            }
        }

        // wait until the loop BA is finished
        while (SLAM.loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }

        // automatically close the viewer
#ifdef USE_PANGOLIN_VIEWER
        if (auto_term) {
            viewer.request_terminate();
        }
#elif USE_SOCKET_PUBLISHER
        if (auto_term) {
            publisher.request_terminate();
        }
#endif
    });

    // run the viewer in the current thread
#ifdef USE_PANGOLIN_VIEWER
    viewer.run();
#elif USE_SOCKET_PUBLISHER
    publisher.run();
#endif

    thread.join();

    // shutdown the SLAM process
    SLAM.shutdown();

    if (eval_log) {
        // output the trajectories for evaluation
        SLAM.save_frame_trajectory("frame_trajectory.txt", "TUM");
        SLAM.save_keyframe_trajectory("keyframe_trajectory.txt", "TUM");
        // output the tracking times for evaluation
        std::ofstream ofs("track_times.txt", std::ios::out);
        if (ofs.is_open()) {
            for (const auto track_time : track_times) {
                ofs << track_time << std::endl;
            }
            ofs.close();
        }
    }

    if (!map_db_path.empty()) {
        // output the map database
        SLAM.save_map_database(map_db_path);
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
}

int main(int argc, char *argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif

    // create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
    // used for mono tracking
    auto img_dir_path = op.add<popl::Value<std::string>>("i", "img-dir", "directory path which contains images");
    auto seg_dir_path = op.add<popl::Value<std::string>>("s", "segmentation-dir", "segmentation directory path");
    // used for stereo tracking. Inferior to mono
    auto img_dir_path_right = op.add<popl::Value<std::string>>("r", "img-dir-right",
                                                               "directory path which contains the right images of a stereo recording");
    auto seg_dir_path_right = op.add<popl::Value<std::string>>("t", "segmentation-dir-right",
                                                               "segmentation directory path for the right images");

    auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
    auto mask_img_path = op.add<popl::Value<std::string>>("", "mask", "mask image path", "");
    auto frame_skip = op.add<popl::Value<unsigned int>>("", "frame-skip", "interval of frame skip", 1);
    auto no_sleep = op.add<popl::Switch>("", "no-sleep", "not wait for next frame in real time");
    auto auto_term = op.add<popl::Switch>("", "auto-term", "automatically terminate the viewer");
    auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
    auto eval_log = op.add<popl::Switch>("", "eval-log", "store trajectory and tracking times for evaluation");
    auto map_db_path = op.add<popl::Value<std::string>>("p", "map-db", "store a map database at this path after SLAM",
                                                        "");
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!vocab_file_path->is_set() || !config_file_path->is_set()) {
        std::cerr << "invalid arguments" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    if (debug_mode->is_set()) {
        spdlog::set_level(spdlog::level::debug);
    } else {
        spdlog::set_level(spdlog::level::info);
    }

    // load configuration
    std::shared_ptr<openvslam::config> cfg;
    try {
        cfg = std::make_shared<openvslam::config>(config_file_path->value());
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStart("slam.prof");
#endif

    // run tracking
    if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Monocular) {
        if (!img_dir_path->is_set() || !seg_dir_path->is_set()) {
            std::cerr << "invalid arguments for mono tracking" << std::endl;
            std::cerr << std::endl;
            std::cerr << op << std::endl;
            return EXIT_FAILURE;
        }

        mono_tracking(cfg, vocab_file_path->value(), img_dir_path->value(), seg_dir_path->value(),
                      mask_img_path->value(),
                      frame_skip->value(), no_sleep->is_set(), auto_term->is_set(),
                      eval_log->is_set(), map_db_path->value());
    } else if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Stereo) {
        if (!img_dir_path->is_set() || !img_dir_path_right->is_set() ||
            !seg_dir_path->is_set() || !seg_dir_path_right->is_set()) {
            std::cerr << "invalid arguments for stereo tracking" << std::endl;
            std::cerr << std::endl;
            std::cerr << op << std::endl;
            return EXIT_FAILURE;
        }

        stereo_tracking(cfg, vocab_file_path->value(), img_dir_path->value(), img_dir_path_right->value(),
                        seg_dir_path->value(), seg_dir_path_right->value(),
                        mask_img_path->value(),
                        frame_skip->value(), no_sleep->is_set(), auto_term->is_set(),
                        eval_log->is_set(), map_db_path->value());
    } else {
        throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif

    return EXIT_SUCCESS;
}
