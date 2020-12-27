#ifndef _INCLUDE_TORSODETECTOR_HPP_
#define _INCLUDE_TORSODETECTOR_HPP_

#include <string>
#include <vector>
#include <memory>          // std::shared_ptr
#include "CVUtils.hpp"
#include "StampedMat.hpp"  // cv::StampedMat

namespace asbgo::vision {

//------------------------------------------------------------------------------
/// @brief      Struct that stores the parametrization for the TorsoDetector class
///
struct TorsoDetectorParameters {
    // ROI computation
    float     roi_width_factor           = 1.5;
    float     roi_height_factor          = 1.5;
    float     roi_bottom_row_factor      = 0.1;
    // legacy params
    float     min_area_threshold         = 100;  // (cf. PostureVision.cpp, ::calculateShoulderKeypoints)
    uint      min_hull_size              = 3;    // (cf. PostureVision.cpp, ::calculateShoulderKeypoints)
    uint      min_defect_distance        = 5;    // 12 -> (cf. PostureVision.cpp, ::calculateShoulderKeypoints) -> too high and nck convities are not detected!
    // foreground extraction
    float     min_depth                  = 0.3;  // 0.6
    float     background_max_depth       = 1.5;
    // blurring/bloating
    uint      blur_kernel_size_1         = 20;
    uint      blur_kernel_size_2         = 20;  // 10
    uint      morph_close_kernel_size    = 20;   // 20; // use the same parameter morphological_kernel_size
    // segmentation
    bool      contour_approximation      = true;
    cv::Point coordinate_offset          = cv::Point(0, 0);  // offset for contour coordinates, useful e.g. when a ROI is being used
    float     min_contour_area           = 0;
    // upper point position estimation
    uint      min_outer_hull_size        = 3;
    float     shoulder_min_height_factor = 0.5;   // added by @joaoandre, throws exception if shoulders detected below min heigh (x nRows)
    // lower point position estimation
    uint      morphological_kernel_size  = 8;
    float     max_hip_height_factor      = 0.2;    // start row to look for hip points, relative to the input frame's height
    bool      use_neck_reference         = false;  // use neck center point instead of shoulder center/midpoint when computing torso center and verifying torso height. @note: if true, reference torso height needs to be increased/ compensated!
    float     torso_center_height_factor = 0.3;    // factor for torso center calculation of midpoint for (0.3 x distance)
    bool      force_hip_width            = true;   // if true, compares reference_hip_width in world coordinates (requires accurate _camera_intrinsics)
    // subject dimensions
    bool      adjust_reference_point     = true;
    float     reference_torso_height     = 0.48;  // subject torso height, in meters/px -> this needs to be a class member, not an algorithm parameter
    float     reference_hip_width        = 0.25;
    float     reference_shoulder_width   = 0.32;  // for ROI estimation
    bool      return_aux_points          = false;
    //--------------------------------------------------------------------------
    const cv::Mat*  camera_intrinsics    = nullptr;
    cv::Mat*  roi_mask                   = nullptr;
    //--------------------------------------------------------------------------
    /// @brief  pointer type alias for convenience
    ///
    /// @note   allows simple & fast changes in smart pointer types if necessary (e.g. std::shared_ptr vs cv::Ptr)
    ///
    typedef cv::Ptr< TorsoDetectorParameters > Ptr;
    //--------------------------------------------------------------------------
    /// @brief      Loads parameters from a YAML/JSON file
    ///
    /// @return     Smart (shared) pointer to a new FeetDetectorParameters instance
    ///
    static Ptr load(const std::string& file);
    //--------------------------------------------------------------------------
    /// @brief      Creates a new instance (heap)
    ///
    /// @return     Smart (shared) pointer to a new FeetDetectorParameters instance
    ///
    static Ptr create() { return Ptr(new TorsoDetectorParameters); }
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------
/// @brief      This class describes a torso detector.
///
/// @todo       Use cv::Pixel alias? for clarity/verbosity
///
class TorsoDetector {
 public:
    //--------------------------------------------------------------------------
    /// @brief   Smart pointer type nested alias for convenience
    ///
    typedef cv::Ptr< TorsoDetector > Ptr;

    //--------------------------------------------------------------------------
    /// @brief      Keypoint enumerator (partial correlation to cv::Skeleton joints)
    ///
    enum Keypoint { LeftNeckEdge, Neck,           RightNeckEdge,
                    LeftShoulder, ShoulderCenter, RightShoulder,
                                  TorsoCenter,
                    LeftHip,      HipCenter,      RightHip };

    //--------------------------------------------------------------------------
    /// @brief      number of points being detected (+ auxiliary points)
    ///
    static const int N_POINTS = 10;

    //--------------------------------------------------------------------------
    /// @brief      Constructs a new instance
    ///
    /// @param[in]  roi_mask           Binary image signaling pixels under analysis
    /// @param[in]  camera_intrinsics  Camera intrinsic matrix
    /// @param[in]  parameters         Torso detection parameters, cf. TorsoDetectorParameters definition
    ///                                Defaults to instance allocated by TorsoDetectorParameters::create()
    ///
    explicit TorsoDetector(const TorsoDetectorParameters::Ptr& parameters = TorsoDetectorParameters::create());

    //--------------------------------------------------------------------------
    /// @brief      Torso detection configuration
    ///
    /// @return     Reference to member configuration (shared pointer type)
    ///
    TorsoDetectorParameters::Ptr& config();

    //--------------------------------------------------------------------------
    /// @brief      Extract torso keypoints/skeleton joints
    ///
    /// @note       For advanced use and/or access to auxilary objects such as foreground/background images and torso contours/convex hull,
    ///             use direct calls to static member functions where image processing is implemented
    ///
    /// @param[in]  frame           input depth image
    ///
    /// @return     vector of keypoints (cv::Point2i):
    ///             [NECK_LEFT, NECK_MID, NECK_RIGHT, SHOULDER_LEFT, SHOULDER_MID, SHOULDER_RIGHT, TORSO_CENTER, HIP_LEFT, HIP_CENTER, HIP_RIGHT]
    ///
    /// @todo       declare a static "detect" function as well?
    ///
    std::vector< cv::Point2i > detect(const cv::Mat& depth_frame, cv::Mat* detection_frame = nullptr);

    //--------------------------------------------------------------------------
    /// @brief      Builds an adjusted rectangular ROI from torso dimensions.
    ///
    /// @param[in]  center_pixel  The center pixel
    /// @param[in]  depth_frame   Initial depth frame / reference frame
    /// @param[in]  parameters    The parameters
    ///
    /// @return     Vector of 2D points (image frame/coordinates) corresponding to ROI corners.
    ///
    static std::vector< cv::Point2i > buildROI(const cv::Point2i& center_pixel, const cv::Mat& depth_frame, const TorsoDetectorParameters::Ptr& parameters = TorsoDetectorParameters::create());
    static std::vector< cv::Point2i > buildTorsoROI(const cv::Mat& depth_frame, const TorsoDetectorParameters::Ptr& parameters = TorsoDetectorParameters::create());

    //--------------------------------------------------------------------------
    /// @brief      Extracts foreground mask (binary image labeling pixels wich are not relevant
    ///             Applies distance thresholdds
    ///
    /// @param[in]  frame            input depth image
    /// @param[in]  parameters       detection parameters
    ///
    /// @return     Binary/depth image (of same type as input depth frame) with background pixels set to null (0) values
    ///
    static cv::Mat extractForeground(const cv::Mat& depth_frame, const TorsoDetectorParameters::Ptr& parameters = TorsoDetectorParameters::create());

    //--------------------------------------------------------------------------
    /// @brief      Smoothing and morphological operations on depth image, with the aim of bloating/dilating a torso mask.
    ///
    /// @param[in]  depth_frame  input depth image
    /// @param[in]  parameters   detection parameters
    ///
    /// @return     Grayscale image (of same type as input depth frame) with smoothed/dilated foreground
    ///
    static cv::Mat bloatTorso(const cv::Mat& depth_frame, const TorsoDetectorParameters::Ptr& parameters);

    //--------------------------------------------------------------------------
    /// @brief      Extracts torso contour from depth image
    ///
    /// @param[in]  depth_frame  input depth image
    /// @param[in]  parameters   detection parameters
    ///
    /// @return     Shape object (aka vector of 2D points) with the largest contour detected on input image frame
    ///
    static cv::Shape2i segmentTorso(const cv::Mat& depth_frame, const TorsoDetectorParameters::Ptr& parameters = TorsoDetectorParameters::create());

    //--------------------------------------------------------------------------
    /// @brief      Locates upper keypoints by looking at the input contours
    ///
    /// @param[in]  torso_contour        torso contour
    /// @param[in]  outer_torso_contour  bloated/outer torso contour
    /// @param[in]  parameters           detection parameters
    ///
    /// @return     Vector of 2D points corresponting to estimated positions of
    ///             [NECK_LEFT, NECK_MID, NECK_RIGHT, SHOULDER_LEFT,
    ///             SHOULDER_MID, SHOULDER_RIGHT]
    ///
    static std::vector< cv::Point2i > locateUpperKeypoints(const cv::Shape2i& torso_contour, const cv::Shape2i& outer_torso_contour, const TorsoDetectorParameters::Ptr& parameters = TorsoDetectorParameters::create());
    static std::vector< cv::Point2i > locateUpperKeypoints2(const cv::Shape2i& torso_contour, const cv::Shape2i& outer_torso_contour, const TorsoDetectorParameters::Ptr& parameters = TorsoDetectorParameters::create());

    //--------------------------------------------------------------------------
    /// @brief      Locates hip keypoints by looking at the input contours
    ///
    /// @param[in]  depth_frame        input depth image
    /// @param[in]  foreground_mask    foreground mask (as extracted w/ ::extractForeground())
    /// @param[in]  shoulder_midpoint  shoulder midpoint/chest center
    /// @param[in]  camera_intrinsics  camera intrinsic matrix
    /// @param[in]  parameters         detection parameters
    ///
    /// @return     Vector of 2D points corresponting to estimated positions of
    ///             [HIP_LEFT, HIP_MID, HIP_RIGHT]
    ///
    static std::vector< cv::Point2i > locateLowerKeypoints(const cv::Mat& depth_frame, const cv::Mat& foreground_mask, const cv::Point& shoulder_midpoint, const TorsoDetectorParameters::Ptr& parameters = TorsoDetectorParameters::create());
    static std::vector< cv::Point2i > locateLowerKeypoints2(const cv::Mat& depth_frame, const cv::Mat& foreground_mask, const std::vector< cv::Point2i >& upper_keypoints, const TorsoDetectorParameters::Ptr& parameters = TorsoDetectorParameters::create());

    //--------------------------------------------------------------------------
    /// @brief      Draws keypoints on image
    ///
    /// @param[in]  frame          input frame (where to draw keypoints) of CV_8UC3 type
    /// @param[in]  keypoints      detected keypoints
    /// @param[in]  invert_sides   invert side flag, drawing left as right joints and vice-versa; defaults to true
    /// @param[in]  draw_segments  draw segments flag, draws line between markers on each foot; defaults to true
    /// @param[in]  draw_aux       draw auxiliary points (neck left/right + chest); defaults to false
    ///
    static void drawKeypoints(cv::Mat& frame, const std::vector< cv::Point2i >& keypoints, bool invert_sides = true, bool draw_segments = true, bool draw_aux = false);

    //--------------------------------------------------------------------------
    /// @brief      Evaluates detection results by comparing them to subject body dimensions.
    ///             Helpful to filter out bad detections/single-frame outliers in sensitive applications. 
    ///
    /// @param[in]  depth_frame  input depth frame (for 3D reconstruction)
    /// @param[in]  keypoints    detected torso keypoints
    /// @param[in]  parameters   detection parameters (w/ reference dimensions)
    ///
    /// @return     floating point normalized quality metric, within [0.0, 1.0].
    ///
    static float detectionQuality(const cv::Mat& depth_frame, const std::vector< cv::Point2i >& keypoints, const TorsoDetectorParameters::Ptr& parameters);

    //--------------------------------------------------------------------------
    /// @brief      Factory method to create new instance on the heap.
    ///
    /// @param[in]  args    Skeleton constructor arguments (as parameter pack).
    ///
    /// @tparam     ARG_TS  Variadic parameter pack encompassing argument types accepted by TorsoDetector constructors
    ///
    /// @return     Smart shared pointer owning new instance created
    ///
    template < typename... ARG_Ts >
    static Ptr create(ARG_Ts... args) { return Ptr(new TorsoDetector(args...)); }

 protected:
    //--------------------------------------------------------------------------
    /// @brief      Algorithm parameter struct.
    ///
    TorsoDetectorParameters::Ptr _parameters;
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace asbgo::vision
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif  // _INCLUDE_TORSODETECTOR_HPP_
