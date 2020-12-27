#include <iostream>
#include <string>
#include <algorithm>  // std::sort
#include <numeric>    // std::accumulate
#include <opencv2/core.hpp>
#include "TrialData.hpp"
#include "CVUtils.hpp"
#include "FeetDetector.hpp"

#define CLOSE_KEY          27   // 'ESC'

using SensorID = asbgo::vision::TrialData::SensorID;
using asbgo::vision::FeetDetector;

//--------------------------------------------------------------------------

int main(int argc, char const *argv[]) {

    //--------------------------------------------------------------------------
    // parse input arguments
    if (argc < 3) {
        throw std::invalid_argument("main(): insufficient arguments; please use ./detectKeypoints [SOURCE_DIR] [ANKLE_HEIGHT]");
    }
    // source directory (trial data)
    std::string root_path(argv[1]);
    // detection params
    float ankle_height= atof(argv[2]);

    // target file (tracked skeleton joint positisons)
    std::cout << "ROOT DIR: "       << root_path      << std::endl;
    std::cout << "ANKLE HEIGHT: "   << ankle_height   << std::endl;

    // init trial data loader
    auto trial_params = asbgo::vision::TrialDataParameters::create();

    asbgo::vision::TrialData trial(root_path, trial_params);

    //--------------------------------------------------------------------------
    // load initial frames, performs initial loader alignment -> there is some overhead due to different start times!
    // @note: asbgo::vision::TrialData loads all sensors, we only need the depth currently
    printf("Aligning initial frames....\n");
    std::vector< cv::StampedMat > frames;

    // this should be done on frame syncrhonizer class
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
    // std::cout << avg_stamp << std::endl

    // method 2: direct from file
    // @todo: write class that averages ovr multiple frames
    cv::Mat floor_reference = cv::imread("config/floor_reference.png", cv::IMREAD_ANYDEPTH);
    if (floor_reference.empty()) {
        throw std::runtime_error("Invalid floor reference file!");
    }
    asbgo::vision::TrialData::descaleFrame(floor_reference);

    // compute camera pitch before cropping to ROI
    // std::vector< cv::Point2i > { cv::Point2i(0.5 * floor_reference.cols, 0.6 * floor_reference.cols), cv::Point2i(0.5 * floor_reference.cols, 0.3 * floor_reference.rows)};
    std::vector< cv::Point2i > threshold_pixels { cv::Point2i(0.6 * floor_reference.rows, 0.5 * floor_reference.cols),
                                                  cv::Point2i(0.3 * floor_reference.rows, 0.5 * floor_reference.cols) };

    double angle = cv::estimateVerticalPitch(floor_reference,
                                             cv::pixelToPoint(threshold_pixels[0]),
                                             cv::pixelToPoint(threshold_pixels[1]),
                                             trial.intrinsics(SensorID::GAIT_RGB));


    std::cout <<  "Estimated camera pitch: " << angle << " rad (" << ((angle * 180) / M_PI) << " º)" << std::endl;


    // create ROI masks
    cv::Mat gait_roi_mask;

    FeetDetector feet_detector;
    feet_detector.config()->ankle_height = ankle_height;
    feet_detector.config()->roi_mask = &gait_roi_mask;
    feet_detector.config()->floor_distance = &floor_reference;
    feet_detector.config()->camera_intrinsics = &trial.intrinsics(SensorID::GAIT_RGB);

    // input key
    int key = -1;
    bool first = true;

    // joint containers
    std::vector< cv::Point2i > feet_joints;

    // cv::VideoWriter gait_recorder("foot_detection.avi", cv::VideoWriter::fourcc('M','J','P','G'), 30, frames[SensorID::GAIT_RGB].size());

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

        // ticks = (double)cv::getTickCount();
        try {
            feet_joints  = feet_detector.detect(frames[SensorID::GAIT_DEPTH], &frames[SensorID::GAIT_RGB]);
            // feet_joints  = feet_detector.detect(frames[SensorID::GAIT_DEPTH]);
        } catch (std::runtime_error& error) {
            std::cout << "Feet detection failure: " << error.what() << std::endl;
        }
        // gait_fps = cv::getTickFrequency() / (double(cv::getTickCount()) - ticks);
        // printf("FPS: %.3f\n", gait_fps);

        //--------------------------------------------------------------------------
        /// @note       left/right sides need to be swapped!
        if (feet_joints.size() > 0) {
            cv::Point3d right_foot = cv::imageToWorld< float >(feet_joints[FeetDetector::Keypoint::LeftAnkle], frames[SensorID::GAIT_DEPTH], trial.intrinsics(SensorID::GAIT_RGB), true, true);
            cv::Point3d right_toe  = cv::imageToWorld< float >(feet_joints[FeetDetector::Keypoint::LeftToe], frames[SensorID::GAIT_DEPTH], trial.intrinsics(SensorID::GAIT_RGB), true, true);
            cv::Point3d reft_foot  = cv::imageToWorld< float >(feet_joints[FeetDetector::Keypoint::RightAnkle], frames[SensorID::GAIT_DEPTH], trial.intrinsics(SensorID::GAIT_RGB), true, true);
            cv::Point3d reft_toe   = cv::imageToWorld< float >(feet_joints[FeetDetector::Keypoint::RightToe], frames[SensorID::GAIT_DEPTH], trial.intrinsics(SensorID::GAIT_RGB), true, true);
        }

        //--------------------------------------------------------------------------
        // cv::imshow("Feet keypoints [DEPTH]", frames[SensorID::GAIT_DEPTH]);
        cv::imshow("Feet keypoints [RGB]", frames[SensorID::GAIT_RGB]);

        //--------------------------------------------------------------------------
        // gait_recorder << frames[SensorID::GAIT_RGB];

        // //--------------------------------------------------------------------------
        key = cv::waitKey(1);
        if (key == CLOSE_KEY) {
            break;
        }
    }
    // gait_recorder.release();

    return 0;
}
