#include <iostream>
#include <string>
#include <algorithm>
#include <numeric>
#include <opencv2/core.hpp>
#include "TrialData.hpp"
#include "CVUtils.hpp"
#include "TorsoDetector.hpp"

#define CLOSE_KEY 27 // 'ESC'

using SensorID = asbgo::vision::TrialData::SensorID;
using asbgo::vision::TorsoDetector;

//--------------------------------------------------------------------------

int main(int argc, char const *argv[]) {

    //--------------------------------------------------------------------------
    // parse input arguments
    if (argc < 5) {
        std::cout << argc << std::endl;
        throw std::invalid_argument("main(): insufficient arguments; please use ./torsoDetector [SOURCE_DIR] [SHOULDER_WIDTH] [HIP_WIDTH] [TORSO_HEIGHT]");
    }
    // source directory (trial data)
    std::string root_path(argv[1]);
    // detection params
    float shoulder_width = atof(argv[2]);
    float hip_width = atof(argv[3]);
    float torso_height= atof(argv[4]);

    // target file (joint positisons)
    std::cout << "ROOT DIR: "       << root_path      << std::endl;
    std::cout << "SHOULDER WIDTH: " << shoulder_width << std::endl;
    std::cout << "HIP WIDTH: "      << hip_width      << std::endl;
    std::cout << "TORSO HEIGHT: "   << torso_height   << std::endl;

    // init trial data loader
    auto trial_params = asbgo::vision::TrialDataParameters::create();

    asbgo::vision::TrialData trial(root_path, trial_params);

    //--------------------------------------------------------------------------
    // load initial frames, performs initial loader alignment -> there is some overhead due to different start times!
    // @note: asbgo::vision::TrialData loads all sensors, we only need the depth currently
    printf("Aligning initial frames....\n");
    std::vector< cv::StampedMat > frames;

    try {
        try {
            frames = trial.data().next(0.030);
        } catch (std::runtime_error&) {
            std::cout << "Unable to synchronize frame, attempting a bigger threshold!"<< std::endl;
            frames = trial.data().next(0.065);
        }
    } catch (std::runtime_error& err) {
        printf("\n ERROR: Unable to perform initial frame alignment, aborting\n\n");
        return(1);
    }
    asbgo::vision::TrialData::descaleFrame(frames[SensorID::POSTURE_DEPTH]);
    asbgo::vision::TrialData::descaleFrame(frames[SensorID::GAIT_DEPTH]);

    //--------------------------------------------------------------------------
    // compute average time stamp
    // since only depth frames will be used, averaging between both depth images results in a more accurate time stamp for the skeleton
    cv::TimeStamp avg_stamp = 0.5 * frames[SensorID::POSTURE_DEPTH].stamp + 0.5 * frames[SensorID::GAIT_DEPTH].stamp;

    std::cout.precision(6);
    std::cout << "Initial time stamps: " << std::fixed << frames[SensorID::POSTURE_RGB].stamp << ", " << frames[SensorID::POSTURE_DEPTH].stamp << ", "
                                                       << frames[SensorID::GAIT_RGB].stamp    << ", " << frames[SensorID::GAIT_DEPTH].stamp << std::endl;
    // create ROI masks
    cv::Mat torso_roi_mask;

    // detector instances
    TorsoDetector torso_detector;
    torso_detector.config()->reference_torso_height = torso_height;
    torso_detector.config()->reference_hip_width = hip_width;
    torso_detector.config()->reference_shoulder_width = shoulder_width;
    torso_detector.config()->roi_mask = &torso_roi_mask;
    torso_detector.config()->camera_intrinsics = &trial.intrinsics(SensorID::POSTURE_RGB);

    // input key
    int key    = -1;
    bool first = true;

    // joint containers
    std::vector< cv::Point2i > torso_joints;

    // array for error computation
    float   torso_fps;
    double  ticks;  // = (double)cv::getTickCount();

    // cv::VideoWriter posture_recorder("torso_detection.avi", cv::VideoWriter::fourcc('M','J','P','G'), 30, frames[SensorID::POSTURE_RGB].size());

    while (1) {
        //--------------------------------------------------------------------------
        // load frames
        try {
            try {
                frames = trial.data().next(0.030);
            } catch (std::runtime_error&) {
                // if it can't synchronize within 30 ms, try with a highr threshold!
                // @note: even if call to next() raises exception, the process of attempting to synchronize loads new frames!
                std::cout << "Unable to synchronize frame, attempting a bigger threshold!"<< std::endl;
                frames = trial.data().next(0.065);
            }
        } catch (std::runtime_error&) {
            printf("No more synchronized frames!\n");
            break;
        }        
        asbgo::vision::TrialData::descaleFrame(frames[SensorID::POSTURE_DEPTH]);
        asbgo::vision::TrialData::descaleFrame(frames[SensorID::GAIT_DEPTH]);

        //--------------------------------------------------------------------------
        // compute avrage time stamp
        avg_stamp = 0.5 * frames[SensorID::POSTURE_DEPTH].stamp + 0.5 * frames[SensorID::GAIT_DEPTH].stamp;

        //--------------------------------------------------------------------------
        /// update depth detector
        // ticks = (double)cv::getTickCount();
        try {
            torso_joints = torso_detector.detect(frames[SensorID::POSTURE_DEPTH], &frames[SensorID::POSTURE_RGB]);
        } catch (std::runtime_error& error) {
            std::cout << "Torso detection failure: " << error.what() << std::endl;
        }
        // torso_fps = cv::getTickFrequency() / (double(cv::getTickCount()) - ticks);
        // ticks = (double)cv::getTickCount();
        // printf("FPS: %.3f\n", torso_fps);

        //--------------------------------------------------------------------------
        /// assign detected keypoints to skeleton joints
        /// @note   left/right sides need to be swapped!
        cv::Point3d neck            = cv::imageToWorld< float >(torso_joints[TorsoDetector::Keypoint::Neck], frames[SensorID::POSTURE_DEPTH], trial.intrinsics(SensorID::POSTURE_RGB), true, true);
        cv::Point3d left_upper_arm  = cv::imageToWorld< float >(torso_joints[TorsoDetector::Keypoint::RightShoulder], frames[SensorID::POSTURE_DEPTH], trial.intrinsics(SensorID::POSTURE_RGB), true, true);
        cv::Point3d right_upper_arm = cv::imageToWorld< float >(torso_joints[TorsoDetector::Keypoint::LeftShoulder], frames[SensorID::POSTURE_DEPTH], trial.intrinsics(SensorID::POSTURE_RGB), true, true);
        cv::Point3d torso_l5        = cv::imageToWorld< float >(torso_joints[TorsoDetector::Keypoint::TorsoCenter], frames[SensorID::POSTURE_DEPTH], trial.intrinsics(SensorID::POSTURE_RGB), true, true);
        cv::Point3d pelvis          = cv::imageToWorld< float >(torso_joints[TorsoDetector::Keypoint::HipCenter], frames[SensorID::POSTURE_DEPTH], trial.intrinsics(SensorID::POSTURE_RGB), true, true);
        cv::Point3d left_upper_leg  = cv::imageToWorld< float >(torso_joints[TorsoDetector::Keypoint::RightHip], frames[SensorID::POSTURE_DEPTH], trial.intrinsics(SensorID::POSTURE_RGB), true, true);
        cv::Point3d right_upper_leg = cv::imageToWorld< float >(torso_joints[TorsoDetector::Keypoint::LeftHip], frames[SensorID::POSTURE_DEPTH], trial.intrinsics(SensorID::POSTURE_RGB), true, true);

        //--------------------------------------------------------------------------
        // cv::imshow("Torso keypoints [DEPTH]", frames[SensorID::POSTURE_DEPTH]);
        cv::imshow("Torso keypoints [RGB]", frames[SensorID::POSTURE_RGB]);

        //--------------------------------------------------------------------------
        // posture_recorder << frames[SensorID::POSTURE_RGB];

        // //--------------------------------------------------------------------------
        key = cv::waitKey(1);
        if (key == CLOSE_KEY) {
            break;
        }
    }
    // posture_recorder.release();

    return 0;
}
