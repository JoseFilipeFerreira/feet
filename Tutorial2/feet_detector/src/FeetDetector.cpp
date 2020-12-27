#include <vector>
#include <memory>
#include <exception>
#include "FeetDetector.hpp"
#include "StampedMat.hpp"


namespace asbgo::vision {


FeetDetector::FeetDetector(const FeetDetectorParameters::Ptr& parameters) :
    _parameters(parameters) {
        /* ... */
}


FeetDetectorParameters::Ptr& FeetDetector::config() {
    return _parameters;
}


std::vector< cv::Point2i > FeetDetector::detect(const cv::Mat& depth_frame, cv::Mat* detection_frame) {
    //--------------------------------------------------------------------------
    /// replace any NaN value on depth frame with null values, otherwise NaN pixels will polute image processing operations
    ///
    cv::patchNaNs(depth_frame, 0.0);
    //--------------------------------------------------------------------------
    /// build feet ROI (hadcoded)
    ///
    /// @todo   parametrize this (vs hardcoded)
    ///
    if (_parameters->roi_mask == nullptr || _parameters->roi_mask->empty()) {
        // build feet ROI
        // @todo: parametrize this vs hardcoded
        std::vector< cv::Point2i > feet_roi = {cv::Point2f(160, 400),
                                               cv::Point2f(260,  80),
                                               cv::Point2f(400,  80),
                                               cv::Point2f(500, 400)};
        *_parameters->roi_mask = cv::Mat(depth_frame.size(), CV_8UC1, cv::Scalar(255));
        *_parameters->roi_mask = cv::toROI< uint8_t >(*_parameters->roi_mask, feet_roi, false);
        cv::imshow("Feet ROI mask", *_parameters->roi_mask);
        cv::imshow("Floor reference", *_parameters->floor_distance);
        cv::waitKey(1);
        cv::destroyWindow("Floor reference");
        cv::destroyWindow("Feet ROI mask");
        // throw std::invalid_argument("FeetDetector::detect(): invalid/empty ROI mask");
    }
    //--------------------------------------------------------------------------
    if (depth_frame.size() != _parameters->roi_mask->size()) {
        throw std::invalid_argument("FeetDetector::detect(): invalid input frame (size != mask size)");
    }
    //--------------------------------------------------------------------------
    if (depth_frame.size() != _parameters->floor_distance->size()) {
        throw std::invalid_argument("FeetDetector::detect(): invalid floor depth reference frame (size != mask size)");
    }
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    // file output for validation purposes
    // cv::imwrite("figs/feet/original_depth_x10.png", depth_frame * 10);
    //--------------------------------------------------------------------------
    // get foreground_mask
    cv::Mat foreground_mask = extractForeground(depth_frame, _parameters);
    // cv::imshow("lllll", foreground_mask);
    // printf("1. Foreground extracted\n");
    //--------------------------------------------------------------------------
    // find foot contours (blobs)
    std::vector< cv::Shape2i > foot_contours;
    foot_contours = segmentFeet(foreground_mask, _parameters);  // returns 2 contours/blobs&
    // printf("2. Feet segmented (found %d)\n", foot_contours.size());
    //--------------------------------------------------------------------------
    // check segmentation results and estimate keypoints
    std::vector< cv::Point2i > foot_keypoints(N_POINTS);
    if (foot_contours.size() > 0) {
        auto keypoints = locateFootKeypoints(foot_contours[0], _parameters);
        foot_keypoints[LeftAnkle] = keypoints[1];
        foot_keypoints[LeftToe]   = keypoints[0];
    }
    if (foot_contours.size() > 1) {
        auto keypoints = locateFootKeypoints(foot_contours[1], _parameters);
        foot_keypoints[RightAnkle] = keypoints[1];
        foot_keypoints[RightToe]   = keypoints[0];
    }
    // printf("3. Keypoints Located\n");
    //--------------------------------------------------------------------------
    // check if sizes match { left, right }, otherwise swap feet
    // @todo       declare member enum type for feet and keypoint indexing!
    if (foot_keypoints.size() > 1) {
        // chck which foot is to the left (smaller 'x' value)
        if (cv::midpoint2D(foot_keypoints[LeftAnkle], foot_keypoints[LeftToe]).x > cv::midpoint2D(foot_keypoints[RightAnkle], foot_keypoints[RightToe]).x) {
            // swap feet
            cv::Point2i tmp = foot_keypoints[LeftAnkle];
            foot_keypoints[LeftAnkle] = foot_keypoints[RightAnkle];
            foot_keypoints[RightAnkle] = tmp;
            tmp = foot_keypoints[LeftToe];
            foot_keypoints[LeftToe] = foot_keypoints[RightToe];
            foot_keypoints[RightToe] = tmp;
        }
    }
    //--------------------------------------------------------------------------
    // construct output detection image w/ detection results
    // overlays contours and keypoint/skeleton (assumes detection frame is of 8UC3 type)
    if (detection_frame != nullptr) {
        for (uint idx = 0; idx < foot_contours.size(); idx++) {
            // std::vector< int > foot_hull_idx;
            // cv::convexHull(foot_contours[idx], foot_hull_idx, true, false /* return indexes */);
            // cv::Shape2i foot_hull = cv::contourSubset(foot_contours[idx], foot_hull_idx);
            // cv::drawContours(*detection_frame, std::vector< cv::Shape2i >{ foot_hull }, 0, cv::Scalar(0, 100, 100), 2);
            cv::drawContours(*detection_frame, foot_contours, idx, cv::Scalar(0, 100, 100), 2);
        }
        drawKeypoints(*detection_frame, foot_keypoints);
        // cv::imshow("Feet Detection", *detection_frame);
        // cv::waitKey(1);
    }


    //--------------------------------------------------------------------------
    // cv::imwrite("figs/feet/foot_contours_keypoints.png", contour_img);
    // cv::imshow("Foot Blobs", contour_img);
    // cv::waitKey(0);
    return foot_keypoints;
}
//--------------------------------------------------------------------------
/// @brief      Foreground extraction function, labels each pixel as floor or not-floor
///
/// @param      foreground_mask    The foreground mask
/// @param[in]  depth_frame        The depth frame
/// @param[in]  floor_depth_image  The floor depth image
/// @param[in]  parameters         The parameters
///
cv::Mat FeetDetector::extractForeground(const cv::Mat& depth_frame, const FeetDetectorParameters::Ptr& parameters) {
    // check if input frame is single-channel float type
    if (depth_frame.channels() > 1 || depth_frame.depth() != CV_32F) {
        throw std::invalid_argument("FeetDetector::extractForeground(): invalid input depth image type!");
    }
    if (parameters->floor_distance->empty()) {
        throw std::invalid_argument("FeetDetector::extractForeground(): invalid floor reference!");
    }
    if (parameters->floor_distance->size() != depth_frame.size()) {
        throw std::invalid_argument("FeetDetector::extractForeground(): input depth image and floor reference have different sizes!");
    }
    //--------------------------------------------------------------------------
    // initialize foreground mask (overwrites)
    cv::Mat depth_foreground = cv::Mat::zeros(depth_frame.size(), depth_frame.type());
    cv::Mat background_mask  = cv::Mat::zeros(depth_frame.size(), CV_8UC1);
    cv::Mat foreground_mask  = cv::Mat::zeros(depth_frame.size(), CV_8UC1);
    //--------------------------------------------------------------------------
    for (uint row_idx = 0; row_idx < depth_frame.rows; row_idx++) {
        // @note: fastest way to iterate through a cv::Mat is to get row pointers
        const uint8_t* mask_row           = parameters->roi_mask->ptr< uint8_t >     (row_idx);
        const float* depth_row            = depth_frame.ptr< float >                 (row_idx);
        const float* floor_reference_row  = parameters->floor_distance->ptr< float > (row_idx);
        float*       depth_foreground_row = depth_foreground.ptr< float >            (row_idx);
        uint8_t*     background_mask_row  = background_mask.ptr< uint8_t >           (row_idx);
        uint8_t*     foreground_mask_row  = foreground_mask.ptr< uint8_t >           (row_idx);

        for (uint col_idx = 0; col_idx < depth_frame.cols; col_idx++) {
            //--------------------------------------------------------------------------
            // skip pixel if outside ROI
            if (mask_row[col_idx] == 0) {
                background_mask_row[col_idx] = static_cast< uint8_t >(255);
                continue;
            }
            //--------------------------------------------------------------------------
            // compute depth thresholds
            float min_distance = floor_reference_row[col_idx];
            if (parameters->fixed_depth_thresholding == true) {
                // adjust thresholding with higth value (in meters), compensating camera's pitch angle
                min_distance -= ((parameters->ankle_height * parameters->ankle_height_factor) / cos(parameters->camera_pitch_angle));
            } else {
                // set min depth as % of floor reference
                // compensates camera pitch angle, but relies on the quality of the floor reference
                min_distance -= (parameters->min_depth_threshold_factor * floor_reference_row[col_idx]);
            }
            float max_distance = floor_reference_row[col_idx] - (parameters->floor_depth_tolerance * floor_reference_row[col_idx]);
            //--------------------------------------------------------------------------
            // thresholding
            // if value on input pixel closer than [floor_distance_tolerance] to the camera, then the pixel is not floor
            if (depth_row[col_idx] > min_distance && depth_row[col_idx] < max_distance) {
                depth_foreground_row[col_idx] = depth_row[col_idx];
                foreground_mask_row[col_idx] = static_cast< uint8_t >(255);
            } else {
                background_mask_row[col_idx] = static_cast< uint8_t >(255);
            }
        }
    }
    //--------------------------------------------------------------------------
    /// morphological operations
    /// @note:      frame is modified in place
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(parameters->morphological_kernel_size, parameters->morphological_kernel_size));
    cv::morphologyEx(foreground_mask, foreground_mask, cv::MORPH_ERODE,  kernel);
    cv::morphologyEx(foreground_mask, foreground_mask, cv::MORPH_CLOSE,  kernel);
    cv::morphologyEx(foreground_mask, foreground_mask, cv::MORPH_DILATE, kernel);
    return foreground_mask;
}
//------------------------------------------------------------------------------
/// @brief      Finds two largest contours on input foreground mask (binary image)
///             Object segmentation (contouring) following OpenCV's implementation (Border Following)
///
/// @ref        Satoshi Suzuki and others. Topological structural analysis of digitized binary images by border following.
///             Computer Vision, Graphics, and Image Processing, 30(1):32–46, 1985.
//////
/// @param[out] foot_contours    output foot contours to populate
/// @param[in]  foreground_mask  8-bit single channel image to segment
/// @param[in]  parameters       Ptr to feet detector parameters
///
/// @throws     std::invalid_argument   1) if input mask is not a 8-bit binary image
/// @throws     std::runtime_error      1) if less than two contours are found
///                                     2) if less that two contours with area above threshold are found
std::vector< cv::Shape2i > FeetDetector::segmentFeet(const cv::Mat& foreground_mask, const FeetDetectorParameters::Ptr& parameters) {
    if (foreground_mask.type() != CV_8UC1) {
        throw std::invalid_argument("FeetDetector::segmentFeet(): invalid input depth image type!");
    }
    //--------------------------------------------------------------------------
    /// @note: retrieval type modifies how contour hierarchy is populated, RETR_EXTERNAL searchs only for the outer contours
    ///        c.f. https://docs.opencv.org/3.4.7/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
    ///
    std::vector< cv::Shape2i > all_contours;
    cv::findContours(foreground_mask,
                     all_contours,              // foot_contours are used temporarily to store segmentation results to avoid unnecessary copies
                     cv::RETR_EXTERNAL,
                     parameters->contour_approximation ? cv::CHAIN_APPROX_SIMPLE : cv::CHAIN_APPROX_NONE,
                     parameters->coordinate_offset);
    //--------------------------------------------------------------------------
    /// parsing and filtering detected contours
    // if (all_contours.size() < 2) {
    //     throw std::runtime_error("FeetDetector::segmentFeet(): insufficient contours on image (< 2)!");
    // }
    //--------------------------------------------------------------------------
    /// find 2 greatest contours from segmentation results
    std::vector< int > largest_idxs = cv::largestShapeIndex(all_contours, 2, parameters->max_contour_area, parameters->min_contour_area);
    //--------------------------------------------------------------------------
    /// check if at least two contours w/ valid areas were found
    // if (largest_idxs[0] < 0 || largest_idxs[1] < 0) {
    //     throw std::runtime_error("FeetDetector::segmentFeet(): insufficient contours with valid areas (< 2)!");
    // }
    //--------------------------------------------------------------------------
    /// assign contours to output object, only blobs with min area value are returned
    /// therefor a single-blob or empty vector is possible
    std::vector< cv::Shape2i > foot_contours;
    for (const auto& contour_idx : largest_idxs) {
        if (contour_idx != -1) {
            foot_contours.emplace_back(all_contours[contour_idx]);
        }
    }
    //--------------------------------------------------------------------------
    /// return convex contour if parametrized to do so
    /// only computes hull if contours are not already convex!
    if (parameters->force_convex_contours == true) {
        for (auto& foot_contour : foot_contours) {
            if (!cv::isContourConvex(foot_contour)) {
                cv::convexHull(foot_contour, foot_contour);
            }
        }
    }
    //--------------------------------------------------------------------------
    return foot_contours;
}
//------------------------------------------------------------------------------
/// @brief      Estimates foot keypoints (ankle and toe // base and tip) from its contours. Assumes optimal elliptical shape for foot contour.
///
/// @param      ankle         pointer to ankle joint object
/// @param      tip           pointer to toe joint object
/// @param[in]  foot_blob     foot contour as cv::Shape2i aka std::vector< cv::Point2i >
/// @param[in]  parameters    Ptr to feet detector parameters
/// @param      bounding_box  bounding box enclosing optimal-fit ellipse around the contour; provides rotation angle along the vertical axis
/// @param      centroid      centroid computed from contour points, useful to assign left/right foot positions (alternatively, bounding_box center can be used)
///
std::vector< cv::Point2i > FeetDetector::locateFootKeypoints(const cv::Shape2i& foot_contour, const FeetDetectorParameters::Ptr& parameters) {
    // match an ellipse to the contour shape
    // @note       alternatives: fitEllipse(foot_contour) / fitCircle(foot_contour) / fitLine(foot_contour)
    cv::RotatedRect foot_box;
    if (parameters->force_ellipse_over_rect == false) {
        foot_box = cv::minAreaRect(foot_contour);
    } else {
        foot_box = cv::fitEllipse(foot_contour);
    }
    //--------------------------------------------------------------------------
    // get corners of rotated rectangle
    cv::Point2f box_corners[4];
    foot_box.points(box_corners);
    // get midpoint of larger sides of rectangle
    std::vector< cv::Point2i > keypoints(2);
    keypoints[0] = cv::midpoint2D(box_corners[0], box_corners[1]);
    keypoints[1] = cv::midpoint2D(box_corners[2], box_corners[3]);
    if (abs(keypoints[0].x - keypoints[1].x) > abs(keypoints[0].y - keypoints[1].y)) {
        keypoints[0] = cv::midpoint2D(box_corners[0], box_corners[3]);
        keypoints[1] = cv::midpoint2D(box_corners[1], box_corners[2]);
    }

    //--------------------------------------------------------------------------
    // assign contour to the contour point closest to the current estimation
    if (parameters->force_contour_keypoints) {
        for (auto& point : keypoints) {
            int idx = cv::closestPolygonPoint(point, foot_contour);
            point = foot_contour[idx];
        }
    }
    //--------------------------------------------------------------------------
    // assign contour to the contour point closest to the current estimation
    // float slope = (keypoints[1].y - keypoints[0].y) / (keypoints[1].x - keypoints[0].x);
    // double foot_in_contour = 0.0;
    // float step
    // while (foot_in_contour <= 0.0) {
    //     auto pt    = cv::Point2i(keypoints[0].x++, keypoints[0].y - slope);
    //     in_contour = cv::pointPolygonTest(foot_contour, pt, false);
    // }
    // double toe_in_contour = 0.0;
    // while (toe_in_contour <= 0.0) {
    
    // }



    //--------------------------------------------------------------------------
    /// check foot length (world coordinates)
    /// ...
    // if (parameters->check_shoe_length) {
    //
    // }
    //--------------------------------------------------------------------------
    // chech if there is intersection?
    // cv::rotatedRectangleIntersect(bounding_box_0, bounding_box_1, intersect_pixels);
    //--------------------------------------------------------------------------
    return keypoints;
}

void FeetDetector::drawKeypoints(cv::Mat& frame, const std::vector< cv::Point2i >& foot_keypoints, bool invert_sides, bool draw_segment) {
    cv::Scalar skeleton_point_color(0, 150, 150);
    cv::Scalar right_segment_color(255, 0, 0);
    cv::Scalar left_segment_color(0, 0, 255);
    if (foot_keypoints.size() > 0) {
        cv::drawMarker(frame, foot_keypoints[Keypoint::LeftAnkle], skeleton_point_color, cv::MARKER_SQUARE, 10, 1);
        cv::drawMarker(frame, foot_keypoints[Keypoint::LeftToe], skeleton_point_color, cv::MARKER_SQUARE, 10, 1);
        cv::line(frame, foot_keypoints[LeftAnkle], foot_keypoints[LeftToe], left_segment_color, 2);
    }
    if (foot_keypoints.size() > 2) {
        cv::drawMarker(frame, foot_keypoints[Keypoint::RightAnkle], skeleton_point_color, cv::MARKER_SQUARE, 10, 1);
        cv::drawMarker(frame, foot_keypoints[Keypoint::RightToe], skeleton_point_color, cv::MARKER_SQUARE, 10, 1);
        cv::line(frame, foot_keypoints[RightAnkle], foot_keypoints[RightToe], right_segment_color, 2);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace asbgo::vision
