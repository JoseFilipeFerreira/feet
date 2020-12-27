
#include <iostream>
#include <opencv2/core.hpp>
#include "TrialData.hpp"
#include "CVUtils.hpp"
#include "MultiTracker2.hpp"

#define MARKER_RADIUS_PX   10   // 8  // px
#define CLOSE_KEY          27   // 'ESC'
#define RESET_KEY          114  // 'r'
#define NUM_TORSO_TRACKERS 6
#define NUM_FOOT_TRACKERS  4

using SensorID = asbgo::vision::TrialData::SensorID;
using asbgo::vision::TrialData;

//------------------------------------------------------------------------------
/// @brief      Marker identifier index and labels
///
static const std::vector< std::string > TorsoMarker { "Neck", "Left Shoulder", "Right Shoulder", "TorsoL5 (CoM)", "Left Hip", "Right Hip" };
static const std::vector< std::string > FootMarker  { "Left Ankle", "Left Toe", "Right Ankle", "Right Toe" };

//--------------------------------------------------------------------------

int main(int argc, char const *argv[]) {

    // tracker labels (hardcoded cf. cv::Tracker implementation)
    static std::vector< std::string > trackers = {"BOOSTING",
                                                  "MIL",
                                                  "KCF",
                                                  "TLD",
                                                  "MEDIANFLOW",
                                                  "GOTURN",
                                                  "MOSSE",
                                                  "CSRT"};

    // parse input arguments
    if (argc < 3) {
        throw std::invalid_argument("main(): insufficient arguments; please use ./trackMarkers [SOURCE_DIR] [TRACKER_ID]");
    }
    // source directory (trial data)
    std::string root_path(argv[1]);

    // tracker type
    uint tracker_type = atoi(argv[2]);
    if (tracker_type >= trackers.size()) {
        throw std::invalid_argument("main(): invalid tracker type;");
    }

    std::cout << "ROOT DIR: " << root_path              << std::endl;
    std::cout << "TRACKER: "  << trackers[tracker_type] << std::endl;

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
    cv::TimeStamp avg_stamp = 0.25 * frames[SensorID::POSTURE_RGB].stamp + 0.25 * frames[SensorID::POSTURE_DEPTH].stamp +
                              0.25 * frames[SensorID::GAIT_RGB].stamp    + 0.25 * frames[SensorID::GAIT_DEPTH].stamp;

    std::cout.precision(6);
    std::cout << "Initial time stamps: " << std::fixed << frames[SensorID::POSTURE_RGB].stamp << ", " << frames[SensorID::POSTURE_DEPTH].stamp << ", "
                                                       << frames[SensorID::GAIT_RGB].stamp    << ", " << frames[SensorID::GAIT_DEPTH].stamp    << std::endl;

     // multi-tracker instances
    cv::Ptr< cv::MultiTracker2 > torso_marker_tracker = cv::MultiTracker2::create(TorsoMarker.size(), trackers[tracker_type]);
    cv::Ptr< cv::MultiTracker2 > gait_marker_tracker  = cv::MultiTracker2::create(FootMarker.size(), trackers[tracker_type]);

    // init trackers (user-set marker coordinates)
    std::cout << "Please identify torso markers manually, in the order: [Neck, LeftShoulder, RightShoulder, TorsoL5 (CoM), LeftUpperLeg, RightUpperLeg] \n";
    torso_marker_tracker->init(frames[SensorID::POSTURE_RGB], 2 * MARKER_RADIUS_PX);
    std::cout << "Please identify foot markers manually, in the order: [LeftFoot (Ankle), RightFoot (Ankle), LeftToe (Tip), RightToe (Tip)] \n";
    gait_marker_tracker->init(frames[SensorID::GAIT_RGB], 2 * MARKER_RADIUS_PX);

    // fps counters (for debugging)
    float   torso_fps, gait_fps;
    double  ticks;  // = (double)cv::getTickCount();

    // input key
    int key = -1;

    // first loop flag
    bool first = true;

    // success flags for each frame, for each marker
    std::vector< bool > torso_success, gait_success;

    while (1) {
        //--------------------------------------------------------------------------
        // load frames
        try {
            try {
                frames = trial.data().next(0.030);
            } catch (std::runtime_error&) {
                // if it can't synchronize within 30 ms, try with a highr threshold!
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
        avg_stamp = 0.25 * frames[SensorID::POSTURE_RGB].stamp + 0.25 * frames[SensorID::POSTURE_DEPTH].stamp +
                    0.25 * frames[SensorID::GAIT_RGB].stamp    + 0.25 * frames[SensorID::GAIT_DEPTH].stamp;

        //--------------------------------------------------------------------------
        // track markers
        torso_success = torso_marker_tracker->update(frames[SensorID::POSTURE_RGB]);
        gait_success  = gait_marker_tracker->update(frames[SensorID::GAIT_RGB]);

        //--------------------------------------------------------------------------
        // check tracking success for both torso and gait markers
        cv::MultiTracker2::drawMarkers(&frames[SensorID::POSTURE_RGB], torso_marker_tracker->markers(), torso_success);
        cv::MultiTracker2::drawMarkers(&frames[SensorID::GAIT_RGB], gait_marker_tracker->markers(), gait_success);

        //--------------------------------------------------------------------------
        // reset visual markers if tracking failed
        for (int idx = 0; idx < TorsoMarker.size() ; idx++) {
            if (torso_success[idx] == false) {
                printf("Reset marker: %s\n", TorsoMarker[idx].data());
                torso_marker_tracker->init(frames[SensorID::POSTURE_RGB], 2 * MARKER_RADIUS_PX, idx);
            }
        }
        for (int idx = 0; idx < FootMarker.size() ; idx++) {
            if (gait_success[idx] == false) {
                printf("Reset marker: %s\n", FootMarker[idx].data());
                gait_marker_tracker->init(frames[SensorID::GAIT_RGB], 2 * MARKER_RADIUS_PX, idx);
            }
        }

        //--------------------------------------------------------------------------
        // assign skeleton 3D joints to marker positions
        // @note: torso and feet joints use different reference frames!
        // @note: same order needs to be kept when selecting the markers initial box
        // @note   left/right sides need to be swapped!
        // torso joints
        cv::Point3d neck            = cv::imageToWorld< double >(cv::centroid2D(torso_marker_tracker->markers()[0]) /* Neck */,           frames[SensorID::POSTURE_DEPTH], trial.intrinsics(SensorID::POSTURE_RGB));
        cv::Point3d left_upper_arm  = cv::imageToWorld< double >(cv::centroid2D(torso_marker_tracker->markers()[2]) /* Right Shoulder */, frames[SensorID::POSTURE_DEPTH], trial.intrinsics(SensorID::POSTURE_RGB));
        cv::Point3d right_upper_arm = cv::imageToWorld< double >(cv::centroid2D(torso_marker_tracker->markers()[1]) /* Left Shoulder  */, frames[SensorID::POSTURE_DEPTH], trial.intrinsics(SensorID::POSTURE_RGB));
        cv::Point3d torso_L5        = cv::imageToWorld< double >(cv::centroid2D(torso_marker_tracker->markers()[3]) /* Torso (CoM) */,    frames[SensorID::POSTURE_DEPTH], trial.intrinsics(SensorID::POSTURE_RGB));
        cv::Point3d left_upper_leg  = cv::imageToWorld< double >(cv::centroid2D(torso_marker_tracker->markers()[5]) /* Right Hip */,      frames[SensorID::POSTURE_DEPTH], trial.intrinsics(SensorID::POSTURE_RGB));
        cv::Point3d right_upper_leg = cv::imageToWorld< double >(cv::centroid2D(torso_marker_tracker->markers()[4]) /* Left Hip */,       frames[SensorID::POSTURE_DEPTH], trial.intrinsics(SensorID::POSTURE_RGB));
        // gait joints
        cv::Point3d left_foot       = cv::imageToWorld< double >(cv::centroid2D(gait_marker_tracker->markers()[1]) /* Right Foot  */, frames[SensorID::GAIT_DEPTH], trial.intrinsics(SensorID::GAIT_RGB));
        cv::Point3d right_foot      = cv::imageToWorld< double >(cv::centroid2D(gait_marker_tracker->markers()[0]) /* Left Foot */,   frames[SensorID::GAIT_DEPTH], trial.intrinsics(SensorID::GAIT_RGB));
        cv::Point3d left_toe        = cv::imageToWorld< double >(cv::centroid2D(gait_marker_tracker->markers()[3]) /* Right Toe */,   frames[SensorID::GAIT_DEPTH], trial.intrinsics(SensorID::GAIT_RGB));
        cv::Point3d right_toe       = cv::imageToWorld< double >(cv::centroid2D(gait_marker_tracker->markers()[2]) /* Left Toe */,    frames[SensorID::GAIT_DEPTH], trial.intrinsics(SensorID::GAIT_RGB));

        //--------------------------------------------------------------------------
        // show frames
        cv::imshow("Posture", frames[SensorID::POSTURE_RGB]);
        cv::imshow("Gait", frames[SensorID::GAIT_RGB]);

        //--------------------------------------------------------------------------
        key = cv::waitKey(1);
        if (key == CLOSE_KEY) {
            break;
        } else if (key == RESET_KEY) {
            // reset trackers
            std::cout << "Please identify torso markers manually, in the order: [Neck, LeftShoulder, RightShoulder, TorsoL5 (CoM), LeftUpperLeg, RightUpperLeg] \n";
            torso_marker_tracker->init(frames[SensorID::POSTURE_RGB], 2 * MARKER_RADIUS_PX);
            std::cout << "Please identify foot markers manually, in the order: [LeftFoot (Ankle), RightFoot (Ankle), LeftToe (Tip), RightToe (Tip)] \n";
            gait_marker_tracker->init(frames[SensorID::GAIT_RGB], 2 * MARKER_RADIUS_PX);
        }
    }

    return 0;
}