#ifndef _INCLUDE_FEETDETECTOR_HPP_
#define _INCLUDE_FEETDETECTOR_HPP_

#include <vector>
#include <memory>          // std::shared_ptr
#include "CVUtils.hpp"
#include "StampedMat.hpp"  // cv::StampedMat

namespace asbgo::vision {

//------------------------------------------------------------------------------
/// @brief      Struct that stores the parametrization for the FeetDetector class
///
struct FeetDetectorParameters {
    //--------------------------------------------------------------------------
    /// foreground extraction
    ///
    const cv::Mat*  floor_distance       = nullptr;          /*!< reference depth values (distances between camera frame and floor, for each px, wo/ legs/feet) */
    float     camera_pitch_angle         = 0.876198;         /*!< rad, measured as the angle between the camera plane and the floor */
    bool      fixed_depth_thresholding   = true;             /*!< fixed height value used for lower depth thresholding; compensates the camera pitch, but assumes a flat floor with no lateral tilt  */
    float     ankle_height               = 0.1;              /*!< ankle height in meters, subjec-specific */
    float     ankle_height_factor        = 1.7;              /*!< for fixed threshold = ankle_height_factor * ankle_height */
    float     min_depth_threshold_factor = 0.2;              /*!< for non-fixed threshold, min depth = min_depth_threshold_factor * floor_reference */
    float     floor_depth_tolerance      = 0.05;             /*!< reference multiplier for upper depth threshold background/foreground extraction; */
    uint      morphological_kernel_size  = 5;                /*!< kernel size for morphological operations */
    //--------------------------------------------------------------------------
    /// segmentation
    ///
    bool      contour_approximation      = true;             /*!< only looks for top-level (external) contours; faster, less overhead */
    bool      force_convex_contours      = false;            /*!< return contour convex hull after segmentation, instead of segmented shape */
    cv::Point coordinate_offset          = cv::Point(0, 0);  /*!< offset for contour coordinates, useful if input images are ROIs of othr images */
    float     min_contour_area           = 200;              /*!< min contour area for each foot blob candidate */
    float     max_contour_area           = 14000;            /*!< max contour area for each foot blob candidate */
    //--------------------------------------------------------------------------
    /// point localization
    ///
    bool      force_ellipse_over_rect    = false;            /*!< attempts to fit foot blob to an ellipse instead of a rect */
    bool      force_contour_keypoints    = true;             /*!< if true, searches for contour point insted of fitted/estimated rectangle/ellipse side midpoint [ADDS NOISE] */
    bool      check_min_length           = true;             /*!< if true, compares world distance between estimated foot keypoints to subject shoe length */
    //--------------------------------------------------------------------------
    /// foot dimension check
    ///
    bool      force_foot_length          = false;
    float     shoe_length                = 0.27;
    //--------------------------------------------------------------------------
    /// camera intrisic parametrisation for 3D reconstruction
    ///
    const cv::Mat*  camera_intrinsics    = nullptr;
    //--------------------------------------------------------------------------
    /// binary image with pixels to consider for feet detection
    ///
    cv::Mat*        roi_mask             = nullptr;
    //--------------------------------------------------------------------------
    /// @brief  pointer type alias for convenience
    /// 
    /// @note   allows simple & fast changes in smart pointer types if necessary (e.g. std::shared_ptr vs cv::Ptr)
    ///
    typedef cv::Ptr< FeetDetectorParameters > Ptr;
    //--------------------------------------------------------------------------
    /// @brief      Creates a new instance (heap)
    ///
    /// @return     Smart (shared) pointer to a new FeetDetectorParameters instance
    ///
    static Ptr create() { return Ptr(new FeetDetectorParameters); }
};

//------------------------------------------------------------------------------
/// @brief      This class implements two-dimensional detection (as pixel coordinates) of feet keypoints from depth images
///             (Algorithm extensively described in technical report)
///
/// @note       Detection steps/algorithmic implemented in static member functions for a 'bare' interface (wo/ need for instantiaton)
///
class FeetDetector {
 public:
    //--------------------------------------------------------------------------
    /// @brief      Input image type/encoding alias
    ///
    // typedef CV_32FC1 image_t;
    // static const int image_t = CV_32FC1;

    //--------------------------------------------------------------------------
    /// @brief      pointer type alias for convenience
    ///
    typedef cv::Ptr< FeetDetector > Ptr;

    //--------------------------------------------------------------------------
    /// @brief      keypoint enumerator
    ///
    enum Keypoint { LeftAnkle, LeftToe, RightAnkle, RightToe };

    //--------------------------------------------------------------------------
    /// @brief      number of points being detected
    ///
    static const int N_POINTS = 4;

    //--------------------------------------------------------------------------
    /// @brief      Constructs a new instance
    ///
    /// @param[in]  floor_reference    Floor reference frame, a depth image with the distance to floor or fixed objects in the camera' FoV
    /// @param[in]  camera_intrinsics  Intrinsic parametrization of the depth camera
    /// @param[in]  roi_mask           Binary image signaling pixels under analysis
    ///                                Defaults to instance allocated by FeetDetectorParameters::create().
    ///                                A pointer is used to simplify instantiation of multiple detectors using the same parametrization.
    ///
    /// @todo                          Pass only ROI vertices/corners, create mask image within contructor
    ///
    FeetDetector(const FeetDetectorParameters::Ptr& parameters = FeetDetectorParameters::create());

    //--------------------------------------------------------------------------
    /// @brief      Feet detection configuration
    ///
    /// @return     Read-only reference to member configuration (shared pointer type)
    ///
    FeetDetectorParameters::Ptr& config();

    //--------------------------------------------------------------------------
    /// @brief      Extract feet keypoints, wrapping around static member functions.
    ///             Provides an abstraction layer to the detection algorithm, returning foot keypoints directly from input image
    ///
    /// @note       For advanced use and/or access to auxilary objects such as foreground/background images and foot contours,
    ///             use direct calls to static member functions where image processing is implemented
    ///
    /// @param[in]  depth_frame     input depth image

    /// @return     vector of keypoints (cv::Point2i), matching ::Keypoint order:
    ///             [ ANKLE_LEFT, FOOT_LEFT, ANKLE_RIGHT, FOOT_RIGHT ]
    ///
    /// @todo       declare a static "detect" function as well?
    ///
    std::vector< cv::Point2i > detect(const cv::Mat& depth_frame, cv::Mat* detection_frame = nullptr);

    //--------------------------------------------------------------------------
    /// @brief      Constructs foreground mask
    ///
    /// @param[out] foreground_mask   foreground mask (binary image)
    /// @param[in]  depth_frame       input depth image
    /// @param[in]  floor_depth_mask  background mask (binary image)
    /// @param[in]  parameters        detection parameters
    ///
    static cv::Mat extractForeground(const cv::Mat& depth_frame, const FeetDetectorParameters::Ptr& parameters);

    //--------------------------------------------------------------------------
    /// @brief      Finds foot blobs, using OpenCV's implementation (Border Following algorithm)
    ///             Filters contours found on image to the two with greatest areas.
    ///
    /// @param[out] foot_contours    two largest contours found on foreground_mask, no lateral assignment
    /// @param[in]  foreground_mask  foreground mask (binary image), as obtained from extractForeground()
    /// @param[in]  parameters       pointer to parameters structure
    ///
    static std::vector< cv::Shape2i > segmentFeet(const cv::Mat& foreground_mask, const FeetDetectorParameters::Ptr& parameters);

    //--------------------------------------------------------------------------
    /// @brief      Locates keypoints on foot blobs/contours
    ///
    /// @param[out] foot_keypoints  The foot keypoints
    /// @param[in]  foot_blobs      The foot blobs
    /// @param[in]  parameters      The parameters
    ///
    static std::vector< cv::Point2i > locateFootKeypoints(const cv::Shape2i& foot_contour, const FeetDetectorParameters::Ptr& parameters);

    //--------------------------------------------------------------------------
    /// @brief      Draws single foot keypoints
    ///
    /// @param[in]  frame           input frame (where to draw keypoints) of CV_8UC3 type
    /// @param[in]  foot_keypoints  detected keypoints
    /// @param[in]  invert_sides    invert side flag, drawing left as right joints and vice-versa; defaults to true
    /// @param[in]  draw_segments   draw segments flag, draws line between markers on each foot; defaults to true
    ///
    static void drawKeypoints(cv::Mat& frame, const std::vector< cv::Point2i >& foot_keypoints, bool invert_sides = true, bool draw_segments = true);

    //--------------------------------------------------------------------------
    /// @brief      Factory method to create new instance on the heap
    ///
    /// @param[in]  args    Skeleton constructor arguments
    ///
    /// @tparam     ARG_TS  Variadic parameter pack encompassing argument types accepted by FeetDetector constructors
    ///
    /// @return     Smart shared pointer owning new instance created
    ///
    template < typename... ARG_Ts >
    static Ptr create(ARG_Ts... args) { return Ptr(new FeetDetector(args...)); }

 protected:
    //--------------------------------------------------------------------------
    /// @brief      Pointer to configuration instance
    ///
    FeetDetectorParameters::Ptr _parameters;
};

}  // namespace asbgo::vision

#endif  // _INCLUDE_FEETDETECTOR_HPP_
