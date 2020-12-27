#include <string>
#include <memory>
// #include <dirent.h>            // DIR, opendir, closedir
#include "TrialData.hpp"
#include "FrameLoaders.hpp"
#include "FrameSynchronizer.hpp"
#include "CVUtils.hpp"
#include <opencv2/opencv.hpp>     // cv::Ptr, cv::Mat


namespace asbgo::vision {

TrialData::TrialData(const std::string& root_dir, const TrialDataParameters::Ptr& parameters) :
    _id(generateID()),
    _init_stamp(loadSyncStamp(root_dir, parameters)),
    _parameters(parameters),
    _camera_intrinsics(4),
    _camera_extrinsics(cv::Mat::eye(4,4, CV_64FC1)),
    // loaders initialization
    _posture_color_loader_ptr(cv::VideoLoader::create(root_dir + "/" + parameters->posture_rgb_video_file, root_dir + "/" + parameters->posture_rgb_stamps_file)),
    _posture_depth_loader_ptr(cv::ImageLoader::create(root_dir + "/" + parameters->posture_depth_dir, parameters->posture_depth_prefix)),
    _gait_color_loader_ptr(cv::VideoLoader::create(root_dir + "/" + parameters->gait_rgb_video_file, root_dir + "/" + parameters->gait_rgb_stamps_file)),
    _gait_depth_loader_ptr(cv::ImageLoader::create(root_dir + "/" + parameters->gait_depth_dir, parameters->gait_depth_prefix)) {
        // initialize synchronizer / align initial frames
        _sync_loader = cv::FrameSynchronizer({ _posture_color_loader_ptr, _posture_depth_loader_ptr, _gait_color_loader_ptr, _gait_depth_loader_ptr }, 0.050 /* 30 fps samples, max 2 frames threshold! */);
        // load intrinsic matrixes from file paths
        std::vector< std::string > paths { root_dir + "/" + parameters->posture_rgb_intrinsics_path,
                                           root_dir + "/" + parameters->posture_depth_intrinsics_path,
                                           root_dir + "/" + parameters->gait_rgb_intrinsics_path,
                                           root_dir + "/" + parameters->gait_depth_intrinsics_path };

        for (int idx = 0; idx < 4; idx++) {
            cv::FileStorage file(paths[idx], cv::FileStorage::Mode::READ);
            if (file.isOpened()) {
                file["K"] >> _camera_intrinsics[idx];
            } else {
                throw std::runtime_error("TrialData::TrialData(): unable to load intrinsic parametrization file!");
            }            
        }
        // load extrinsic matrix
        cv::FileStorage file(root_dir + "/" + parameters->extrinsics_file, cv::FileStorage::Mode::READ);
        if (file.isOpened()) {
            file["CamGaitToPostureTransform"][0][0] >> _camera_extrinsics.at< double >(0, 0);
            file["CamGaitToPostureTransform"][0][1] >> _camera_extrinsics.at< double >(0, 1);
            file["CamGaitToPostureTransform"][0][2] >> _camera_extrinsics.at< double >(0, 2);
            file["CamGaitToPostureTransform"][0][3] >> _camera_extrinsics.at< double >(0, 3);
            file["CamGaitToPostureTransform"][1][0] >> _camera_extrinsics.at< double >(1, 0);
            file["CamGaitToPostureTransform"][1][1] >> _camera_extrinsics.at< double >(1, 1);
            file["CamGaitToPostureTransform"][1][2] >> _camera_extrinsics.at< double >(1, 2);
            file["CamGaitToPostureTransform"][1][3] >> _camera_extrinsics.at< double >(1, 3);
            file["CamGaitToPostureTransform"][2][0] >> _camera_extrinsics.at< double >(2, 0);
            file["CamGaitToPostureTransform"][2][1] >> _camera_extrinsics.at< double >(2, 1);
            file["CamGaitToPostureTransform"][2][2] >> _camera_extrinsics.at< double >(2, 2);
            file["CamGaitToPostureTransform"][2][3] >> _camera_extrinsics.at< double >(2, 3);
            // no need to assign last line, mat was initialized to identity
        } else {
            // throw std::runtime_error("TrialData::TrialData(): unable to load extrinsic parametrization file!");
        }
        // assign new ID
        // ...
}

TrialData::ID TrialData::id() const {
    return _id;
}

const TrialDataParameters::Ptr& TrialData::parameters()  const {
    return _parameters;
}

const cv::Mat& TrialData::intrinsics(SensorID sensor) const {
    return _camera_intrinsics[sensor];
}

const cv::Mat& TrialData::extrinsics() const {
    return _camera_extrinsics;
}

const cv::TimeStamp& TrialData::initStamp() const {
    return _init_stamp;
}

cv::FrameSynchronizer& TrialData::data() {
    return _sync_loader;
}

/// @todo: implement this
TrialData::ID TrialData::generateID() {
    /* ... */
    return static_cast< TrialData::ID >(0);
}

void TrialData::descaleFrame(cv::Mat& depth_frame, float factor) {
    if (depth_frame.type() != CV_16UC1) {
        throw std::invalid_argument("TrialData::fixDepth(): invalid input frame (type != CV_16UC1)");
    }
    depth_frame.convertTo(depth_frame, CV_32F);
    depth_frame = depth_frame / 1000.0;
}

cv::TimeStamp TrialData::loadSyncStamp(const std::string& path, const TrialDataParameters::Ptr& parameters) {
    // list files on directory
    // DIR *dir;
    // struct dirent *ent;
    // std::string stamp_file_name;
    // if ((dir = opendir (path.data())) != NULL) {
    //     while ((ent = readdir(dir)) != NULL) {
    //         std::string entry(ent->d_name);
    //         if (entry.find(parameters->sync_file_prefix) != std::string::npos && entry.find(parameters->sync_file_extension)) {
    //             std::cout << entry << std::endl;
    //             stamp_file_name = entry;
    //             break;
    //         }
    //     }
    //     closedir(dir);
    // } else {
    //     throw std::runtime_error("loadSyncStamp() -> could not open path;");
    // }

    // make use of ImageLoader::listFiles
    std::vector< std::string > stamp_files = cv::ImageLoader::listFiles(path, parameters->sync_file_extension);
    if (stamp_files.size() == 0) {
        throw std::runtime_error("TrialData::loadSyncStamp(): no sync file found!");
    }
    // use first stamp file by default, ignore remaining if multiple stamp files
    int stamp_start_pos = stamp_files[0].find(parameters->sync_file_prefix) + parameters->sync_file_prefix.size();
    int stamp_mid_pos = stamp_files[0].substr(stamp_start_pos).find(parameters->sec_nsec_separator) + stamp_start_pos;
    int stamp_end_pos = stamp_files[0].find(parameters->sync_file_extension);

    if ((stamp_mid_pos - stamp_start_pos) != parameters->sec_length) {
        // ...
    }
    if ((stamp_end_pos - stamp_mid_pos + 1) != parameters->sec_length) {
        // ...
    }

    int sec = atoi(stamp_files[0].substr(stamp_start_pos, stamp_mid_pos - stamp_start_pos).data());
    int nsec = atoi(stamp_files[0].substr(stamp_mid_pos + 1, stamp_end_pos - stamp_mid_pos + 1).data());
    cv::TimeStamp stamp = static_cast< long double >(sec) + static_cast< long double >(0.000000001 * nsec);
    // std::cout.precision(9);
    // std::cout << "stamp " << std::fixed << stamp << std::endl;

    return stamp;
}

}  // namespace asbgo::vision
