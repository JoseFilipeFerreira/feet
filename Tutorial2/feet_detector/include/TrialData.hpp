#ifndef _INCLUDE_TRIALDATA_HPP_
#define _INCLUDE_TRIALDATA_HPP_

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>     // cv::Ptr, cv::Mat
#include "FrameSynchronizer.hpp"  // cv::FrameSynchronizer
#include "FrameLoaders.hpp"       // cv::ImageLoader, cv::VideoLoader
#include "StampedMat.hpp"         // cv::StampedMat

//------------------------------------------------------------------------------
/// Classes specific for use with the WalkIt smart walker hardware, thus asbgo::vision namespace is used
///
namespace asbgo::vision {
//------------------------------------------------------------------------------
/// @brief      Simple parameter struct for use with TrialData class
///             Describes how data is strutured/organized within the trial's data folder i.e. relevant paths
///
/// @note       compile w/ --std=c++11, since default value assignment on struct definition is only legal since C++11
///
struct TrialDataParameters {
    //--------------------------------------------------------------------------
    std::string posture_rgb_intrinsics_path   = "camera_info/posture_rgb/camera_info.xml";    /*!< file with ROS camera parametrization (sensor_msgs::CameraInfo) for posture RGB camera */
    std::string posture_depth_intrinsics_path = "camera_info/posture_depth/camera_info.xml";  /*!< file with ROS camera parametrization (sensor_msgs::CameraInfo) for posture depth sensor */
    std::string gait_rgb_intrinsics_path      = "camera_info/gait_rgb/camera_info.xml";       /*!< file with ROS camera parametrization (sensor_msgs::CameraInfo) for posture RGB camera */
    std::string gait_depth_intrinsics_path    = "camera_info/gait_depth/camera_info.xml";     /*!< file with ROS camera parametrization (sensor_msgs::CameraInfo) for posture depth sensor */
    std::string posture_rgb_video_file        = "posture_rgb.avi";                            /*!< video file with RGB frames captured with posture camera */
    std::string posture_rgb_stamps_file       = "posture_rgb_time_stamps.txt";                /*!< text file with individual time stamps (format derived from other parameters) for posture RGB video */
    std::string posture_depth_dir             = "posture_depth_registered";                   /*!< directory with depth frames (individual PNG files) captured with posture camera */
    std::string posture_depth_prefix          = "posture_depth_registered";                   /*!< prefix for depth frames PNG files ('[prefix]_[timestamp].png') */
    std::string gait_rgb_video_file           = "gait_rgb.avi";                               /*!< video file with RGB frames captured with gait camera */
    std::string gait_rgb_stamps_file          = "gait_rgb_time_stamps.txt";                   /*!< text file with individual time stamps (format derived from other parameters) for gait RGB video */
    std::string gait_depth_dir                = "gait_depth_registered";                      /*!< directory with depth frames (individual PNG files) captured with the gait camera */
    std::string gait_depth_prefix             = "gait_depth_registered";                      /*!< prefix for depth frames PNG files ('[prefix]_[timestamp].png') */
    std::string sync_file_prefix              = "sync_";                                      /*!< prefix for the external synchronization file ('[prefix]_[timestamp].stamp') */
    std::string sync_file_extension           = ".stamp";                                     /*!< extension of the external synchronization file, defaults to ".stamp" */
    std::string sec_nsec_separator            = "_";                                          /*!< separator string between sec and nsec on depth image files */
    std::string extrinsics_file               = "extrinsics.json";                            /*!< file where extrinsic gait->posture camera matrix is stored */
    int         index_length                  = 5;                                            /*!< number of digits in the file names specifying frame index */
    int         sec_length                    = 10;                                           /*!< number of digits in the file names specifying time stamp sec count */
    int         nsec_length                   = 9;                                            /*!< number of digits in the file names specifying time stamp nanosec count */
    bool        enable_posture_color          = true;                                         /*!< enable/disable flag for posture color stream */
    bool        enable_posture_depth          = true;                                         /*!< enable/disable flag for posture depth stream */
    bool        enable_gait_color             = true;                                         /*!< enable/disable flag for gait color stream */
    bool        enable_gait_depth             = true;                                         /*!< enable/disable flag for gait depth stream */
    //--------------------------------------------------------------------------
    /// @brief  pointer type alias for convenience
    /// 
    /// @note   allows simple & fast changes in smart pointer types if necessary (e.g. std::shared_ptr vs cv::Ptr)
    ///
    typedef cv::Ptr< TrialDataParameters > Ptr;
    //--------------------------------------------------------------------------
    /// @brief      Creates a new instance (heap)
    ///
    /// @return     Smart (shared) pointer to a new TrialDataParameters instance
    ///
    static Ptr create() { return Ptr(new TrialDataParameters); }
};

//------------------------------------------------------------------------------
/// @brief      This class describes vision data captured during a walking trial conducted on the WalkIt smart walker.
///             The vision apparatus consists of two identical depth cameras oriented towards the user's torso and feet, respectively.
///             A TrialData instance allows to interact with vision data (individual time-stamped and synchronized color and depth frames,
///             as well as intrisic and extrinsic parametrization) captured during a particular trial.
///
class TrialData {
 public:
    //--------------------------------------------------------------------------
    /// @brief      Public type, provided for convenience and code readability
    ///
    typedef uint ID;

    //--------------------------------------------------------------------------
    /// @brief      Public enumerator type, describing different sensors/cameras
    ///
    enum SensorID { POSTURE_RGB, POSTURE_DEPTH, GAIT_RGB, GAIT_DEPTH };

    //--------------------------------------------------------------------------
    /// Class constructor
    ///
    /// @param[in]  root_dir    Trial's root path, where color videos, depth frames, and time stamp information are stored.
    /// @param[in]  parameters  Trial data parameters, cf. TrialDataParameters definition
    ///                         Defaults to instance allocated by TrialDataParameters::create()
    ///
    explicit TrialData(const std::string& root_dir, const TrialDataParameters::Ptr& parameters = TrialDataParameters::create());

    //--------------------------------------------------------------------------
    /// @brief      Trial identifier
    ///
    /// @return     Unique identfier value, of nested public type TrialData::ID.
    ///
    ID id() const;

    //--------------------------------------------------------------------------
    /// @brief      Trial data parameters, describing the directory structure within the root path for the instance.
    ///
    /// @return     Smart (shared) pointer to TrialDataParameters instance.
    ///
    const TrialDataParameters::Ptr& parameters() const;

    //--------------------------------------------------------------------------
    /// @brief      Camera intrisic matrix, for 3D/2D projectin/reconstruction (pixel to world coordinates)
    //
    /// @param[in]  sensor  Sensor/camera enumerator/identifier, of nested public type cf. TrialData::SensorID
    ///
    /// @return     Numerical floating-point matrix (type 32FC1) of size 3x3
    ///
    const cv::Mat& intrinsics(SensorID sensor) const;

    //--------------------------------------------------------------------------
    /// @brief      Extrinsic matrix describing a 3D transformation from the bottom (gait) camera to the top (posture) camera
    ///
    /// @return     Numerical floating-point matrix (type 32FC1) of size 4x4
    ///
    const cv::Mat& extrinsics() const;

    //--------------------------------------------------------------------------
    /// @brief      Trial's initial time stamp, used for syncronization with external reference systems (e.g. XSens)
    ///             Loaded from the synchronization file in the trial's root path.
    ///
    /// @return     Time stamp value of type cv::TimeStamp
    ///
    const cv::TimeStamp& initStamp() const;

    //--------------------------------------------------------------------------
    /// @brief      Trial data (color and depth frames) acessor. Returns a reference to a cv::FrameSynchronizer object, which
    ///             can be used to load frames individually or together (synchronized). Cf. cv::FrameSynchronizer definition.
    ///
    /// @return     const reference to a cv::FrameSynchronizer instance
    ///
    cv::FrameSynchronizer& data();

    //--------------------------------------------------------------------------
    // @brief      Generates an unique ID for a trial. Cf. TrialID definition.
    //
    // @return     TrialID objct with unique identifier.
    //
    static ID generateID();

    //--------------------------------------------------------------------------
    /// @brief      Converts frames from 16U to 32F
    ///             Useful to revert depth frames back to floating point values (meters), as they can only be save as 16bit integers
    ///
    /// @param      frame  Frame to be scaled (must be of CV_16UC1 type)
    ///
    static void descaleFrame(cv::Mat& frame, float factor = 1000);

    //--------------------------------------------------------------------------
    /// @brief      Loads external synchronization time stamp.
    ///
    /// @param[in]  path        trial root dir
    /// @param[in]  parameters  trial parameters
    ///
    /// @return     time stamp
    ///
    static cv::TimeStamp loadSyncStamp(const std::string& path, const TrialDataParameters::Ptr& parameters);

 protected:
    //--------------------------------------------------------------------------
    /// unique identifier member
    ///
    const ID _id;

    //--------------------------------------------------------------------------
    /// intrinsic matrixes for each camera stream
    ///
    std::vector< cv::Mat > _camera_intrinsics;

    //--------------------------------------------------------------------------
    /// extrisinc gait->posture camera reference frame transformation
    ///
    cv::Mat _camera_extrinsics;

    //--------------------------------------------------------------------------
    /// time stamp for external synchronization (accurate session start time stamp)
    ///
    const cv::TimeStamp _init_stamp;

    //--------------------------------------------------------------------------
    /// trial parameters (directory/file structure), cf. TrialDataParameters definition
    ///
    /// @note   pointer type allows sharing the parametrization between different objects
    ///
    TrialDataParameters::Ptr _parameters;

    //--------------------------------------------------------------------------
    /// synchronizer instance (fetches frames with similar (< threshold) timestamps) cf. cv::FrameSynchronizer definition
    ///
    cv::FrameSynchronizer _sync_loader;

    //--------------------------------------------------------------------------
    /// individual posture/gait RGB/depth frame loaders (AVI video/PNG files)
    ///
    /// @note       there is no particular need for them to be members, as FrameSynchronizer objects retain ownership of each loader through a shared pointer
    /// @note       cv::Ptr is used for legacy reasons, as classes within cv:: namespace use it as smart pointer type 
    ///
    cv::Ptr< cv::VideoLoader > _posture_color_loader_ptr;
    cv::Ptr< cv::ImageLoader > _posture_depth_loader_ptr;
    cv::Ptr< cv::VideoLoader > _gait_color_loader_ptr;
    cv::Ptr< cv::ImageLoader > _gait_depth_loader_ptr;
};

}  // namespace asbgo::vision

#endif  // _INCLUDE_TRIALDATA_HPP_
