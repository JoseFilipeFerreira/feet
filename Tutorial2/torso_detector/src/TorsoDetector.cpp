#include "TorsoDetector.hpp"
#include <algorithm>


namespace asbgo::vision {


TorsoDetectorParameters::Ptr TorsoDetectorParameters::load(const std::string& path) {
    cv::FileStorage file(path, cv::FileStorage::Mode::READ);
    //...
    return TorsoDetectorParameters::create();
}


TorsoDetector::TorsoDetector(const TorsoDetectorParameters::Ptr& parameters) :
    _parameters(parameters) {
        /* ... */
}


TorsoDetectorParameters::Ptr& TorsoDetector::config() {
    return _parameters;
}


std::vector< cv::Point2i > TorsoDetector::detect(const cv::Mat& depth_frame, cv::Mat* detection_frame) {
    // remove NaN values from input frame (replace with null depth!)
    // @note: this should be done on ROS wrappers, as it is a problem specific to ROS astra package
    cv::patchNaNs(depth_frame, 0.0);
    // check if roi_mask is empty, if so then build new roi mask
    if (_parameters->roi_mask == nullptr || _parameters->roi_mask->empty()) {
        // auto mask = cv::Mat::ones(depth_frame.size(), CV_8UC1);
        *_parameters->roi_mask = cv::Mat(depth_frame.size(), CV_8UC1, 255);
        cv::toROI< uint8_t >(_parameters->roi_mask, *_parameters->roi_mask, buildTorsoROI(depth_frame, _parameters), false);
        // cv::toROI< uint8_t >(_parameters->roi_mask, *_parameters->roi_mask, buildROI(cv::Point2i(0.5 * depth_frame.cols, 0.5 * depth_frame.rows), depth_frame, _parameters), false);
        // throw std::invalid_argument("TorsoDetector::detect(): invalid/empty ROI mask");
    }
    // check if input frame fits mask
    if (depth_frame.size() != _parameters->roi_mask->size()) {
        throw std::invalid_argument("TorsoDetector::detect(): invalid input frame (size != mask size)");
    }
    // check if valid depth frame
    if (depth_frame.channels() > 1 || depth_frame.depth() != CV_32F) {
        throw std::invalid_argument("TorsoDetector::detect(): invalid input depth image type");
    }

    // 1. threshold depth frame
    cv::Mat foreground_mask = extractForeground(depth_frame, _parameters);

    // 2. bloat returned foreground mask
    cv::Mat smoothed_foreground = bloatTorso(foreground_mask, _parameters);

    // // debug, validation purposes
    // cv::imwrite("figs/torso/original_depth_x10.png", 10*depth_frame);
    // cv::imshow("Foreground", foreground_mask);
    // cv::imwrite("media/torso/foreground_mask.png", foreground_mask);
    // cv::imshow("Smoothed_foreground", smoothed_foreground);
    // cv::imwrite("media/torso/smoothed_foreground_mask.png", smoothed_foreground);
    // cv::waitKey(0);

    // 3. find contours of both smoothed (bloted) torso mask and input non-filtered image
    cv::Shape2i smoothed_contours = segmentTorso(smoothed_foreground, _parameters);
    cv::Shape2i contours          = segmentTorso(foreground_mask, _parameters);

    // debug, for validation
    // cv::Mat contour_img = cv::Mat::zeros(foreground_mask.size(), CV_8UC3);
    // cv::drawContours(contour_img, std::vector< cv::Shape2i > {contours}, 0, cv::Scalar(100, 100, 100));
    // cv::drawContours(contour_img, std::vector< cv::Shape2i > {smoothed_contours}, 0, cv::Scalar(0, 0, 255));
    // cv::imshow("Countours", contour_img);
    // // cv::imwrite("figs/torso/original_and_outer_contours.png", contour_img);
    // cv::waitKey(1);

    // 4. extract upper (neck, shoulders, and chest center)
    std::vector< cv::Point2i > torso_keypoints = locateUpperKeypoints2(contours, smoothed_contours, _parameters);

    // 4. extract lower (hip and pelvis) and append to first vector
    std::vector< cv::Point2i > hip_keypoints = locateLowerKeypoints2(depth_frame, foreground_mask, torso_keypoints, _parameters);

    // concatenate point vector @ end of upper point vector
    // small vectors, thus cheap operation, and allows keeping upper and lower point detection implementations separate
    // @note: no need to call std::vector<>::reserve before calling std::vector<>::insert()
    torso_keypoints.insert(torso_keypoints.end(), hip_keypoints.begin(), hip_keypoints.end());
    // torso_keypoints.emplace_back(hip_keypoints[0]);  // TorsoCenter
    // torso_keypoints.emplace_back(hip_keypoints[1]);  // LeftHip
    // torso_keypoints.emplace_back(hip_keypoints[2]);  // Hip Midpoint (Pelvis)
    // torso_keypoints.emplace_back(hip_keypoints[3]);  // RightHip

    //--------------------------------------------------------------------------
    // construct output detection image w/ detection results
    // overlays contours and keypoint/skeleton (assumes detection frame is of 8UC3 type)
    if (detection_frame != nullptr) {
        cv::drawContours(*detection_frame,  std::vector< cv::Shape2i > { contours }, 0, cv::Scalar(0, 100, 100), 2);
        drawKeypoints(*detection_frame, torso_keypoints);
        // cv::imshow("Torso Detection", *detection_frame);
        // cv::waitKey(1);r
    }

    // move ROI
    *_parameters->roi_mask = cv::Mat(depth_frame.size(), CV_8UC1, 255);
    cv::toROI< uint8_t >(_parameters->roi_mask, *_parameters->roi_mask, buildROI(cv::contourCentroid(smoothed_contours), depth_frame, _parameters), false);

    // cv::Mat keypoint_img;
    // std::vector< cv::Mat > channels (3, foreground_backup);
    // cv::merge(channels, keypoint_img);
    // for (const auto& point : torso_keypoints) {
    //     std::cout << point << std::endl;
    //     cv::drawMarker(keypoint_img, point, cv::Scalar(0, 0, 255), cv::MARKER_SQUARE, 10, 2);
    // }
    // // cv::imshow("Estimated Markers", keypoint_img);
    // cv::imwrite("figs/torso/estimated_markers.png", keypoint_img);
    // cv::waitKey(0);

    return torso_keypoints;
}


std::vector< cv::Point2i > TorsoDetector::buildROI(const cv::Point2i& center_pixel, const cv::Mat& depth_frame, const TorsoDetectorParameters::Ptr& parameters) {
    // find optimal width (px from center col)
    uint width_px = 20;
    cv::Point3f left_midpoint  = cv::imageToWorld< float >(cv::Point2i(center_pixel.x - width_px, center_pixel.y), 0.9 /* 2M from camera, ~ normal remote position */, *parameters->camera_intrinsics, true, true);
    cv::Point3f right_midpoint = cv::imageToWorld< float >(cv::Point2i(center_pixel.x + width_px, center_pixel.y), 0.9 /* 2M from camera, ~ normal remote position */, *parameters->camera_intrinsics, true, true);
    float horizontal_distance = abs(right_midpoint.x - left_midpoint.x);
    while (horizontal_distance < parameters->roi_width_factor * parameters->reference_shoulder_width) {  // && px_from_center < 0.5 * depth_frame.cols) {
        width_px++;
        // std::cout << "h: " << horizontal_distance << std::endl;
        left_midpoint  = cv::imageToWorld< float >(cv::Point2i(center_pixel.x - width_px, center_pixel.y), 0.9, *parameters->camera_intrinsics, true, true);
        right_midpoint = cv::imageToWorld< float >(cv::Point2i(center_pixel.x + width_px, center_pixel.y), 0.9, *parameters->camera_intrinsics, true, true);
        horizontal_distance = abs(right_midpoint.x - left_midpoint.x);
    }
    // find optimal height (px from center row)
    uint height_px = 10;
    cv::Point3f lower_midpoint = cv::imageToWorld< float >(cv::Point2i(center_pixel.x, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows), 0.9, *parameters->camera_intrinsics, true, true);
    cv::Point3f upper_midpoint = cv::imageToWorld< float >(cv::Point2i(center_pixel.x, center_pixel.y - height_px), 0.9, *parameters->camera_intrinsics, true, true);
    float vertical_distance = abs(upper_midpoint.y - lower_midpoint.y);
    while (vertical_distance < parameters->roi_height_factor * parameters->reference_torso_height) {  // && - height_px > 0) {
        height_px++;
        // std::cout << "v: " << vertical_distance << std::endl;
        cv::Point3f lower_midpoint = cv::imageToWorld< float >(cv::Point2i(center_pixel.x, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows), 0.9, *parameters->camera_intrinsics, true, true);
        cv::Point3f upper_midpoint = cv::imageToWorld< float >(cv::Point2i(center_pixel.x, center_pixel.y - height_px), 0.9, *parameters->camera_intrinsics, true, true);
        vertical_distance = abs(upper_midpoint.y - lower_midpoint.y);
    }
    // populate output vector
    std::vector< cv::Point2i > corners;
    corners.emplace_back(center_pixel.x - width_px, center_pixel.y - height_px);
    corners.emplace_back(center_pixel.x - width_px, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows);
    corners.emplace_back(center_pixel.x + width_px, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows);
    corners.emplace_back(center_pixel.x + width_px, center_pixel.y - height_px);
    // for (const auto& pt : corners) {
    //     std::cout << pt << std::endl;
    // }
    return corners;
}


std::vector< cv::Point2i > TorsoDetector::buildTorsoROI(const cv::Mat& depth_frame, const TorsoDetectorParameters::Ptr& parameters) {
    // find optimal width (px from center col)
    uint px_from_center = 20;
    cv::Point3f left_side  = cv::imageToWorld< float >(cv::Point2i(0.5 * depth_frame.cols - px_from_center, 0.5 * depth_frame.rows), depth_frame, *parameters->camera_intrinsics, true, true);
    cv::Point3f right_side = cv::imageToWorld< float >(cv::Point2i(0.5 * depth_frame.cols + px_from_center, 0.5 * depth_frame.rows), depth_frame, *parameters->camera_intrinsics, true, true);
    float horizontal_distance = abs(right_side.x - left_side.x);
    while (horizontal_distance < parameters->roi_width_factor * parameters->reference_shoulder_width && px_from_center < 0.5 * depth_frame.cols) {
        px_from_center++;
        left_side  = cv::imageToWorld< float >(cv::Point2i(0.5 * depth_frame.cols - px_from_center, 0.5 * depth_frame.rows), depth_frame, *parameters->camera_intrinsics, true, true);
        right_side = cv::imageToWorld< float >(cv::Point2i(0.5 * depth_frame.cols + px_from_center, 0.5 * depth_frame.rows), depth_frame, *parameters->camera_intrinsics, true, true);
        horizontal_distance = abs(right_side.x - left_side.x);
    }
    // find optimal height (px from center row)
    uint height_px = 20;
    cv::Point3f lower_midpoint = cv::imageToWorld< float >(cv::Point2i(0.5 * depth_frame.cols, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows), depth_frame, *parameters->camera_intrinsics, true, true);
    cv::Point3f upper_midpoint = cv::imageToWorld< float >(cv::Point2i(0.5 * depth_frame.cols, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows - height_px), depth_frame, *parameters->camera_intrinsics, true, true);
    float vertical_distance = abs(upper_midpoint.y - lower_midpoint.y);
    while (vertical_distance < parameters->roi_height_factor * parameters->reference_torso_height && (depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows - height_px) > 0) {
        height_px++;
        cv::Point3f lower_midpoint = cv::imageToWorld< float >(cv::Point2i(0.5 * depth_frame.cols, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows), depth_frame, *parameters->camera_intrinsics, true, true);
        cv::Point3f upper_midpoint = cv::imageToWorld< float >(cv::Point2i(0.5 * depth_frame.cols, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows - height_px), depth_frame, *parameters->camera_intrinsics, true, true);
        vertical_distance = abs(upper_midpoint.y - lower_midpoint.y);
    }
    // populate output vector
    std::vector< cv::Point2i > corners;
    corners.emplace_back(0.5 * depth_frame.cols - px_from_center, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows);
    corners.emplace_back(0.5 * depth_frame.cols - px_from_center, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows - height_px); //height_factor * depth_frame.rows);
    corners.emplace_back(0.5 * depth_frame.cols + px_from_center, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows - height_px); //height_factor * depth_frame.rows);
    corners.emplace_back(0.5 * depth_frame.cols + px_from_center, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows);
    // for (const auto& pt : corners) {
    //     std::cout << pt << std::endl;
    // }
    // corners.emplace_back(0, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows);
    // corners.emplace_back(0, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows - height_px); //height_factor * depth_frame.rows);
    // corners.emplace_back(depth_frame.cols, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows - height_px); //height_factor * depth_frame.rows);
    // corners.emplace_back(depth_frame.cols, depth_frame.rows - parameters->roi_bottom_row_factor * depth_frame.rows);
    return corners;
}


cv::Mat TorsoDetector::extractForeground(const cv::Mat& depth_frame, const TorsoDetectorParameters::Ptr& parameters) {
    // check if input frame fits member mask
    if (depth_frame.size() != parameters->roi_mask->size()) {
        throw std::invalid_argument("FeetDetector::detect(): invalid input frame (size != mask size)");
    }
    // check if valid depth frame
    if (depth_frame.channels() > 1 || depth_frame.depth() != CV_32F) {
        throw std::invalid_argument("FeetDetector::extractForeground(): invalid input depth image type!");
    }
    cv::Mat foreground_mask  = cv::Mat::zeros(depth_frame.size(), CV_8UC1);
    // threshold depth value for all pixels on input frame
    for (uint row_idx = 0; row_idx < depth_frame.rows; row_idx++) {
        const uint8_t* mask_row            = parameters->roi_mask->ptr< uint8_t > (row_idx);
        const float*   depth_row           = depth_frame.ptr< float >             (row_idx);
        uint8_t*       foreground_mask_row = foreground_mask.ptr< uint8_t >       (row_idx);
        for (uint col_idx = 0; col_idx < depth_frame.cols; col_idx++) {
            if (mask_row[col_idx] != 0 && depth_row[col_idx] > parameters->min_depth && depth_row[col_idx] < parameters->background_max_depth) {
                foreground_mask_row[col_idx] = static_cast< uint8_t >(255);
            }
        }
    }
    return foreground_mask;
}


cv::Mat TorsoDetector::bloatTorso(const cv::Mat& depth_frame, const TorsoDetectorParameters::Ptr& parameters) {
    // smoothing (blurring)
    cv::Mat smoothed_depth_frame;
    cv::blur(depth_frame, smoothed_depth_frame, cv::Size(parameters->blur_kernel_size_1, parameters->blur_kernel_size_1));
    cv::blur(depth_frame, smoothed_depth_frame, cv::Size(parameters->blur_kernel_size_2, parameters->blur_kernel_size_2));
    // morphological operations
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(parameters->morph_close_kernel_size, parameters->morph_close_kernel_size));
    // closing
    cv::morphologyEx(smoothed_depth_frame, smoothed_depth_frame, cv::MORPH_CLOSE, kernel);
    return smoothed_depth_frame;
}


cv::Shape2i TorsoDetector::segmentTorso(const cv::Mat& depth_frame, const TorsoDetectorParameters::Ptr& parameters) {
    // check input frame
    if (depth_frame.channels() > 1) {
        throw std::invalid_argument("TorsoDetector::segmentTorso() : invalid input depth frame (channels > 1)");
    }
    // convert to 8 bits if necessary
    if (depth_frame.depth() != CV_8U) {
        // depth_frame.convertTo(CV_8UC1);
        throw std::invalid_argument("TorsoDetector::segmentTorso() : invalid input depth frame (depth != CV_8U)");
    }
    // segmentation (wrap around cv:findContours)
    std::vector< cv::Shape2i > contours;
    cv::findContours(depth_frame,
                     contours,
                     cv::RETR_EXTERNAL,
                     parameters->contour_approximation ? cv::CHAIN_APPROX_SIMPLE : cv::CHAIN_APPROX_NONE,
                     parameters->coordinate_offset);
    // find largest contour & return
    // @todo: write generic function to get largest  N contours from a vector!
    uint  largest_contour_idx = 0;
    float largest_area = 0;
    float contour_area = 0;
    for (uint idx = 0; idx < contours.size(); idx++)  {
        contour_area = cv::contourArea(contours[idx]);
        if (contour_area < parameters->min_contour_area) {
            continue;
        }
        if (contour_area > largest_area) {
            largest_area = contour_area;
            largest_contour_idx = idx;
        }
    }
    // check if any valid contour was found
    if (largest_contour_idx == -1) {
        throw std::runtime_error("TorsoDetector::segmentTorso() : no torso-like valid contour on depth image!");
    }
    // return largest contour
    return contours[largest_contour_idx];
}


std::vector< cv::Point2i > TorsoDetector::locateUpperKeypoints(const cv::Shape2i& torso_contour, const cv::Shape2i& outer_torso_contour, const TorsoDetectorParameters::Ptr& parameters) {
    // compute convex hull and convexity defects
    // cv::Shape2i        outer_torso_hull;
    // cv::convexHull(outer_torso_contour, outer_torso_hull, true, true  /* return points  */);
    std::vector< int > outer_torso_hull_indexes;
    cv::convexHull(outer_torso_contour, outer_torso_hull_indexes, true, false /* return indexes */);

    // create convex hull from indexes, to avoid calling cv::convexHull twice in a row
    cv::Shape2i outer_torso_hull = cv::contourSubset(outer_torso_contour, outer_torso_hull_indexes);

    // check validity of resulting hull
    if (outer_torso_hull_indexes.size() <= parameters->min_outer_hull_size) {
        throw std::runtime_error("TorsoDetector::locateUpperKeypoints(): invalid outer torso convex hull (size <= 3pts)");
    }

    // compute convexity defect
    // @note: defects -> [start_index, end_index, farthest_pt_index, fixpt_depth] on the original contour
    //        fixpt_depth is fixed-point approximation (with 8 fractional bits) of the distance between the farthest contour point and the hull.
    //        That is, to get the floating-point value of the depth will be fixpt_depth/256.0.
    std::vector< cv::Vec4i > defects;
    cv::convexityDefects(outer_torso_contour, outer_torso_hull_indexes, defects);

    // check if at least two mismatch zones were found
    if (defects.size() < 2) {
        throw std::runtime_error("TorsoDetector::locateUpperKeypoints(): too few concavities found on torso (< 2)");
    }

    // loop over concavities, find which have their fartherst points at lower 'y' values (higher in the image -> neck concavities)
    int min_y_defect_idx_1 = -1;
    int min_y_defect_idx_2 = -1;
    for (int idx = 0; idx < defects.size(); idx++) {
        // get distance between convex hull and contour
        float distance_to_hull = defects[idx][3] / 256.0;
        // get index of farthest point (apex of concavity)
        int farthest_point_idx = defects[idx][2];
        // filter by distance to contour
        // ignores small differences between contour and convex hull caused by noise or hardware limitations
        if (distance_to_hull < parameters->min_defect_distance) {
            continue;
        }
        // skip min check on first two passes
        if (min_y_defect_idx_1 < 0) {
            min_y_defect_idx_1 = idx;
            continue;
        } else if (min_y_defect_idx_2 < 0) {
            min_y_defect_idx_2 = idx;
            continue;
        }
        // compare farthest point vertical position with previous highest defect vrtical position
        int min_y_1 = outer_torso_contour[defects[min_y_defect_idx_1][2]].y;
        int min_y_2 = outer_torso_contour[defects[min_y_defect_idx_2][2]].y;
        if (outer_torso_contour[farthest_point_idx].y < min_y_1) {
            // downgrade previous max idx
            min_y_defect_idx_2 = min_y_defect_idx_1;
            min_y_defect_idx_1 = idx;
        } else if (outer_torso_contour[farthest_point_idx].y < min_y_2) {
            min_y_defect_idx_2 = idx;
        }
    }

    // check if few or no highest defects were found (possible due to max distance filtering)
    if (min_y_defect_idx_1 == -1 || min_y_defect_idx_2 == -1) {
        throw std::runtime_error("TorsoDetector::locateUpperKeypoints(): not enough valid convexity defects above distance threshold");
    } else if (min_y_defect_idx_2 == -1) {
        // assign second max y defect as the max -> is this correct?
        min_y_defect_idx_2 = min_y_defect_idx_1;
        // throw std::runtime_error("TorsoDetector::locateUpperKeypoints(): not enough valid convexity defects above distance threshold");
    }

    // instantiation for code verbosity, not necessary (copies cv:Vec4i instance)
    cv::Vec4i neck_concavity_idx_1 = defects[min_y_defect_idx_1];
    cv::Vec4i neck_concavity_idx_2 = defects[min_y_defect_idx_2];

    // // visual feedback (draw outr contour and fill in black to enhance convexity defects)
    cv::Mat contour_img = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::drawContours(contour_img, std::vector< cv::Shape2i >{ outer_torso_hull }, 0, cv::Scalar(0, 0, 150), -1);
    cv::drawContours(contour_img, std::vector< cv::Shape2i >{ outer_torso_contour }, 0, cv::Scalar(150, 0, 0), -1);
    cv::drawContours(contour_img, std::vector< cv::Shape2i >{ cv::contourSubset(outer_torso_contour, neck_concavity_idx_1) }, 0, cv::Scalar(0, 150, 150), 2);
    cv::drawContours(contour_img, std::vector< cv::Shape2i >{ cv::contourSubset(outer_torso_contour, neck_concavity_idx_2) }, 0, cv::Scalar(0, 150, 150), 2);

    // cv::drawMarker(contour_img, cv::centroid2D(outer_torso_contour), cv::Scalar(100, 100, 100), cv::MARKER_TILTED_CROSS, 40, 3);
    cv::drawMarker(contour_img, cv::contourCentroid(outer_torso_contour), cv::Scalar(200, 200, 200), cv::MARKER_CROSS, 40, 3);

    
    cv::imshow("Convex Hull & Convexity Defects", contour_img);
    // cv::imwrite("figs/torso/convex_hull_and_neck_concavities.png", contour_img);
    cv::waitKey(1);

    // neck keypoints on **outer contour/convex hull**
    cv::Point2i outer_neck_left;
    cv::Point2i outer_neck_right;
    cv::Point2i outer_shoulder_left;
    cv::Point2i outer_shoulder_right;

    // compare 'x' coordinate of concavities & assign left and right keypoints!
    // @note: points are either ordered clockwise or counterclockwise. left/right shoulder start > end, and vice versa
    // alternative: compute centroids of {start_idx, end_idx, far_idx } for each neck concavity
    if (outer_torso_contour[neck_concavity_idx_1[2]].x < outer_torso_contour[neck_concavity_idx_2[2]].x) {
        // 1 -> left,  2-> right
        outer_neck_left  = outer_torso_contour[neck_concavity_idx_1[2]];
        outer_neck_right = outer_torso_contour[neck_concavity_idx_2[2]];
        // compare start and end points of each concavity
        // lowest 'y' value -> shoulder on outer contour
        cv::Point2i start_pt_1 = outer_torso_contour[neck_concavity_idx_1[0]];
        cv::Point2i end_pt_1   = outer_torso_contour[neck_concavity_idx_1[1]];
        cv::Point2i start_pt_2 = outer_torso_contour[neck_concavity_idx_2[0]];
        cv::Point2i end_pt_2   = outer_torso_contour[neck_concavity_idx_2[1]];
        outer_shoulder_left  = (start_pt_1.y > end_pt_1.y) ? start_pt_1 : end_pt_1;
        outer_shoulder_right = (start_pt_2.y > end_pt_2.y) ? start_pt_2 : end_pt_2;
    } else {
        // 1 -> right, 2-> left
        outer_neck_right = outer_torso_contour[neck_concavity_idx_1[2]];
        outer_neck_left  = outer_torso_contour[neck_concavity_idx_2[2]];
        // compare start and end points of each concavity
        // lowest 'y' value -> shoulder on outer contour
        cv::Point2i start_pt_1 = outer_torso_contour[neck_concavity_idx_1[0]];
        cv::Point2i end_pt_1   = outer_torso_contour[neck_concavity_idx_1[1]];
        cv::Point2i start_pt_2 = outer_torso_contour[neck_concavity_idx_2[0]];
        cv::Point2i end_pt_2   = outer_torso_contour[neck_concavity_idx_2[1]];
        outer_shoulder_right = (start_pt_1.y > end_pt_1.y) ? start_pt_1 : end_pt_1;
        outer_shoulder_left  = (start_pt_2.y > end_pt_2.y) ? start_pt_2 : end_pt_2;
    }

    // compute central midpoints
    cv::Point neck_center     = cv::midpoint2D(outer_neck_left, outer_neck_right);
    cv::Point shoulder_center = cv::midpoint2D(outer_shoulder_left, outer_shoulder_right);

    // @note: torso contour is not convex!
    cv::Point neck_left = torso_contour[cv::closestPolygonPoint(outer_neck_left, torso_contour)];
    cv::Point neck_right = torso_contour[cv::closestPolygonPoint(outer_neck_right, torso_contour)];
    cv::Point shoulder_left = torso_contour[cv::closestPolygonPoint(outer_shoulder_left, torso_contour)];
    cv::Point shoulder_right = torso_contour[cv::closestPolygonPoint(outer_shoulder_right, torso_contour)];

    return std::vector< cv::Point >({ neck_left,     neck_center,     neck_right,
                                      shoulder_left, shoulder_center, shoulder_right });
}


std::vector< cv::Point2i > TorsoDetector::locateUpperKeypoints2(const cv::Shape2i& torso_contour, const cv::Shape2i& outer_torso_contour, const TorsoDetectorParameters::Ptr& parameters) {
    // compute convex hull and convexity defects
    // cv::Shape2i        outer_torso_hull;
    // cv::convexHull(outer_torso_contour, outer_torso_hull, true, true  /* return points  */);
    std::vector< int > outer_torso_hull_indexes;
    cv::convexHull(outer_torso_contour, outer_torso_hull_indexes, true, false /* return indexes */);

    // create convex hull from indexes, to avoid calling cv::convexHull twice in a row
    cv::Shape2i outer_torso_hull = cv::contourSubset(outer_torso_contour, outer_torso_hull_indexes);

    // check validity of resulting hull
    if (outer_torso_hull_indexes.size() <= parameters->min_outer_hull_size) {
        throw std::runtime_error("TorsoDetector::locateUpperKeypoints(): invalid outer torso convex hull (size <= 3)");
    }

    // compute convexity defect
    // @note: defects -> [start_index, end_index, farthest_pt_index, fixpt_depth] on the original contour
    //        fixpt_depth is fixed-point approximation (with 8 fractional bits) of the distance between the farthest contour point and the hull.
    //        That is, to get the floating-point value of the depth will be fixpt_depth/256.0.
    std::vector< cv::Vec4i > concavities;
    cv::convexityDefects(outer_torso_contour, outer_torso_hull_indexes, concavities);

    // check if at least two mismatch zones were found
    if (concavities.size() < 2) {
        throw std::runtime_error("TorsoDetector::locateUpperKeypoints(): too few concavities found on torso (< 2)");
    }

    // // loop over concavities, find which have their fartherst points at lower 'y' values (higher in the image -> neck concavities)
    // int concavity_idx_1 = -1;
    // int concavity_idx_2 = -1;
    // for (int idx_1 = 0; idx_1 < concavities.size(); idx_1++) {
    //     for (int idx_2 = 0; idx_2 < concavities.size(); idx_2++) {
    //         // metric that quantifies likelihood of a pair of concavities to be neck
    //         // weighs distance to hull proportionally (bigger size -> higher k) and 'y' coordinates inversely (higher on the image -> higher k)
    //         // needs testing only works if all trms are positive, which should happen anyway with row/col indexes
    //         float k = ((concavities[idx_1][3] / 256.0) + (concavities[idx_2][3] / 256.0)) /
    //                    (outer_torso_contour[concavities[idx_1][2]].y + outer_torso_contour[concavities[idx_2][2]].y);
    //     }
    // }

    // loop over concavities
    // under the assumption that neck and armpit concavities are the greater concavities (*if ROI is properly built*)
    int neck_concavity_idx_left  = -1;
    int neck_concavity_idx_right = -1;
    float max_distance_left  = 0.0;
    float max_distance_right = 0.0;
    // @todo       parametrize this!
    // cv::Point2f center = cv::centroid2D(outer_torso_contour);
    cv::Point2f center = cv::contourCentroid(outer_torso_contour);
    for (int idx = 0; idx < concavities.size(); idx++) {
        // get distance between convex hull and contour
        float distance = concavities[idx][3] / 256.0;
        // filter by distance to contour
        // ignores small differences between contour and convex hull caused by noise or hardware limitations
        if (distance < parameters->min_defect_distance) {
            continue;
        }
        // skip concavities below centroid 'y' (bottom half)
        // @note may cause problems if concavity extends too far down (along the arm)
        // @todo parametrize this!
        if (outer_torso_contour[concavities[idx][0]].y > center.y &&
            outer_torso_contour[concavities[idx][1]].y > center.y &&
            outer_torso_contour[concavities[idx][2]].y > center.y) {
            continue;
        }
        // check upper left quadrant (left neck/shoulder concavity)
        // start, end and farthest points of concavity *must* be within top left quadrant
        if (outer_torso_contour[concavities[idx][0]].x < center.x &&
            outer_torso_contour[concavities[idx][1]].x < center.x &&
            outer_torso_contour[concavities[idx][2]].x < center.x) {
            // ...
            if (distance > max_distance_left) {
                max_distance_left = distance;
                neck_concavity_idx_left = idx;
            }
            continue;
        }
        // check upper right quadrant (right neck/shoulder concavity)
        // start, end and farthest points of concavity *must* be within top right quadrant
        if (outer_torso_contour[concavities[idx][0]].x > center.x &&
            outer_torso_contour[concavities[idx][1]].x > center.x &&
            outer_torso_contour[concavities[idx][2]].x > center.x) {
            // ...
            if (distance > max_distance_right) {
                max_distance_right = distance;
                neck_concavity_idx_right = idx;
            }
            continue;
        }
    }
    if (neck_concavity_idx_left < 0 || neck_concavity_idx_right < 0) {
        throw std::runtime_error("TorsoDetector::locateUpperKeypoints(): not enough valid convexity defects above distance threshold");
    }
    // instantiation for code verbosity, not necessary (copies cv:Vec4i instance)
    cv::Vec4i neck_concavity_left  = concavities[neck_concavity_idx_left];
    cv::Vec4i neck_concavity_right = concavities[neck_concavity_idx_right];

    // // visual feedback (draw outr contour and fill in black to enhance convexity defects)
    cv::Mat contour_img = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::drawContours(contour_img, std::vector< cv::Shape2i >{ outer_torso_hull }, 0, cv::Scalar(0, 0, 150), -1);
    cv::drawContours(contour_img, std::vector< cv::Shape2i >{ outer_torso_contour }, 0, cv::Scalar(150, 0, 0), -1);
    cv::drawContours(contour_img, std::vector< cv::Shape2i >{ cv::contourSubset(outer_torso_contour, neck_concavity_left) }, 0, cv::Scalar(0, 150, 150), 2);
    cv::drawContours(contour_img, std::vector< cv::Shape2i >{ cv::contourSubset(outer_torso_contour, neck_concavity_right) }, 0, cv::Scalar(0, 150, 150), 2);

    // cv::drawMarker(contour_img, cv::centroid2D(outer_torso_contour), cv::Scalar(100, 100, 100), cv::MARKER_TILTED_CROSS, 40, 3);
    cv::drawMarker(contour_img, cv::contourCentroid(outer_torso_contour), cv::Scalar(200, 200, 200), cv::MARKER_CROSS, 40, 3);
    cv::imshow("Convex Hull & Convexity Defects", contour_img);
    cv::waitKey(1);

    // assign neck points on *outer* torso contour
    const cv::Point2i& outer_neck_left  = outer_torso_contour[neck_concavity_left[2]];
    const cv::Point2i& outer_neck_right = outer_torso_contour[neck_concavity_right[2]];

    // @note: points are either ordered clockwise or counterclockwise; left/right shoulder start > end, and vice versa
    // initialize references to start & end points for code readability
    const cv::Point2i& start_pt_left  = outer_torso_contour[neck_concavity_left[0]];
    const cv::Point2i& end_pt_left    = outer_torso_contour[neck_concavity_left[1]];
    const cv::Point2i& start_pt_right = outer_torso_contour[neck_concavity_right[0]];
    const cv::Point2i& end_pt_right   = outer_torso_contour[neck_concavity_right[1]];

    // assign shoulder points on *outer* torso contour
    // compare start and end points of each concavity: lowest 'y' value -> shoulder on outer contour
    const cv::Point2i& outer_shoulder_left  = (start_pt_left.y > end_pt_left.y)   ? start_pt_left  : end_pt_left;
    const cv::Point2i& outer_shoulder_right = (start_pt_right.y > end_pt_right.y) ? start_pt_right : end_pt_right;

    // @todo alocate vector to expected size, populate points directly using member enum
    // compute central midpoints
    cv::Point neck_center     = cv::midpoint2D(outer_neck_left, outer_neck_right);
    cv::Point shoulder_center = cv::midpoint2D(outer_shoulder_left, outer_shoulder_right);

    // find closest contour points on torso contour
    // @note: torso contour is not convex!
    cv::Point neck_left      = torso_contour[cv::closestPolygonPoint(outer_neck_left, torso_contour)];
    cv::Point neck_right     = torso_contour[cv::closestPolygonPoint(outer_neck_right, torso_contour)];
    cv::Point shoulder_left  = torso_contour[cv::closestPolygonPoint(outer_shoulder_left, torso_contour)];
    cv::Point shoulder_right = torso_contour[cv::closestPolygonPoint(outer_shoulder_right, torso_contour)];

    return std::vector< cv::Point2i >({ neck_left,     neck_center,     neck_right,
                                        shoulder_left, shoulder_center, shoulder_right });
}


std::vector< cv::Point2i > TorsoDetector::locateLowerKeypoints(const cv::Mat& depth_frame, const cv::Mat& foreground_mask, const cv::Point& reference_upper_point, const TorsoDetectorParameters::Ptr& parameters) {
    if (foreground_mask.type() != CV_8UC1) {
        throw std::invalid_argument("TorsoDetector::locateLowerKeypoints(): invalid input foreground mask");
    }

    // parse upper (shoulder/neck) reference
    // in case of no depth info at reference point, null coordinates imply measuring distance to camera,
    // which leads to bad hip line values
    // finds closest non-null point if parametrized to do so
    cv::Point2i torso_reference;
    if (depth_frame.at< float >(reference_upper_point) == 0.0 && parameters->adjust_reference_point == true) {
        try {
            torso_reference = cv::closestValidPoint< float >(reference_upper_point, depth_frame, 10, 10);   /// @todo: parametrize the search window
        } catch (std::runtime_error&) {
            // throw std::invalid_argument("TorsoDetector::locateLowerKeypoints(): invalid upper reference point (no depth value)");
            torso_reference = reference_upper_point;
        }
    } else {
        torso_reference = reference_upper_point;
    }

    // apply morphological filters to depth foreground
    cv::Mat filtered_foreground_mask = foreground_mask;
    // // morphological operations
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(parameters->morphological_kernel_size, parameters->morphological_kernel_size));
    // // closing (frame is modified in place)
    cv::morphologyEx(filtered_foreground_mask, filtered_foreground_mask, cv::MORPH_CLOSE,  kernel);
    cv::morphologyEx(filtered_foreground_mask, filtered_foreground_mask, cv::MORPH_DILATE, kernel);
    cv::morphologyEx(filtered_foreground_mask, filtered_foreground_mask, cv::MORPH_CLOSE,  kernel);
    cv::morphologyEx(filtered_foreground_mask, filtered_foreground_mask, cv::MORPH_CLOSE,  kernel);

    //--------------------------------------------------------------------------
    // swep over image from the bottom -> up
    // start at a % of frame heigh to avoid hip points on bottom row
    int lowest_torso_row_idx = -1;
    int row_idx = filtered_foreground_mask.rows - 1;
    while (lowest_torso_row_idx == -1 && row_idx > 0) {
        // find lowest row with mask pixels
        uint8_t* foreground_mask_row = filtered_foreground_mask.ptr< uint8_t >(row_idx--);
        for (uint col_idx = 0; col_idx < filtered_foreground_mask.cols; col_idx++) {
            if (foreground_mask_row[col_idx] == 255) {
                // std::cout << foreground_mask.at<uint8_t>(0,filtered_foreground_mask.rows - 1) << std::endl;
                lowest_torso_row_idx = row_idx;
                break;
            }
        }
    }

    // reuse row_idx (reinitialize)
    row_idx = lowest_torso_row_idx - static_cast< int >(parameters->max_hip_height_factor * filtered_foreground_mask.rows);

    //--------------------------------------------------------------------------
    // for each row under analysis:
    // 1. find edge pixels (mask == 1 w/ neighbours == 0)
    // 2. compute distances between consecutive edge pixels, find max value and assing left/right hip to both edges
    // 3. compute world distance between hip midpoint and shoulder midpoint
    // 4. repeat 1-3 until vertical distance overcomes reference torso height (subject specific)

    float torso_height = 0.0;
    cv::Point left_hip;
    cv::Point right_hip;
    cv::Point hip_center;

    // @todo: use parameters->torso_height with a tolerance value to compensate neck height when parameters->use_neck_reference == true?
    while (torso_height < parameters->reference_torso_height && row_idx < lowest_torso_row_idx) {
        // identify edge points on hip row (with neighbours == 0)
        // ignores first and last col on image (assumed hip edges are not there)
        std::vector< uint > mask_hip_row_edge_cols;
        uint8_t* foreground_mask_row = filtered_foreground_mask.ptr< uint8_t >(row_idx);
        for (uint col_idx = 1; col_idx < filtered_foreground_mask.cols - 1; col_idx++) {
            // if mask point with a non-mask neighbour, append to vector
            if (foreground_mask_row[col_idx] > 0 && (foreground_mask_row[col_idx - 1] == 0 || foreground_mask_row[col_idx + 1] == 0)) {
                  mask_hip_row_edge_cols.emplace_back(col_idx);
            }
        }
        if (mask_hip_row_edge_cols.size() < 2) {
            row_idx++;
            continue;
        }
        // from edge poits, calculate horizontal distance in px between consecutive points
        // @note: could be done directly without creating an edge vector, but code would not be as readable? small vector, not expensive
        uint max_width = 0;
        uint left_hip_edge_col;
        uint right_hip_edge_col;
        for (uint idx = 1; idx < mask_hip_row_edge_cols.size(); idx++) {
            uint width_px =  mask_hip_row_edge_cols[idx] - mask_hip_row_edge_cols[idx - 1];
            if (width_px > max_width) {
                max_width = width_px;
                left_hip_edge_col  = mask_hip_row_edge_cols[idx-1];
                right_hip_edge_col = mask_hip_row_edge_cols[idx];
            }
        }
        // compute hip midpoint
        left_hip  = cv::Point(left_hip_edge_col, row_idx);
        right_hip = cv::Point(right_hip_edge_col, row_idx);
        hip_center = cv::midpoint2D(left_hip, right_hip);

        // if empty point (no depth), skip row
        if (depth_frame.at< float >(hip_center) == 0) {
            printf("NULL DEPTH!\n");
            row_idx++;
            continue;
        }

        // average hip row depth values
        float avg = 0.0;
        for (int col_idx = left_hip_edge_col; col_idx <= right_hip_edge_col; col_idx++) {
            avg += depth_frame.at< float >(row_idx, col_idx);
        }
        avg /= (right_hip_edge_col - left_hip_edge_col);
        // std::cout << "avg depth: "    << avg << std::endl;
        // std::cout << "center depth: " << depth_frame.at< float >(hip_center) << std::endl;

        // update torso measured height (3D distance between hip and shoulder midpoints)
        cv::Point3f hip_center_3D = cv::imageToWorld< float >(hip_center, depth_frame, *parameters->camera_intrinsics);
        cv::Point3f reference_3D  = cv::imageToWorld< float >(reference_upper_point, depth_frame, *parameters->camera_intrinsics);
        // cv::Point3f reference_3D  = cv::imageToWorld< float >(torso_reference, depth_frame, *parameters->camera_intrinsics);
        torso_height              = cv::distance3D(hip_center_3D, reference_3D);

        printf("%f < %f depth refrence: %f depth hip: %f\n", torso_height, parameters->reference_torso_height, depth_frame.at< float >(reference_upper_point), depth_frame.at< float >(hip_center));
        // printf("%d < %d\n", row_idx, lowest_torso_row_idx);
        // if (torso_height < parameters->reference_torso_height) printf("TRUE!\n");

        row_idx++;
    }
    printf("--------------\n");

    // std::cout << "torso height: " << torso_height << " (< " << parameters->reference_torso_height << ")"<< std::endl;
    // std::cout << "depth value at reference point: " << depth_frame.at< float >(reference_upper_point) << std::endl;
    // std::cout << "reference point: " << reference_upper_point << std::endl;
    // try {
    //     if (depth_frame.at< float >(reference_upper_point) == 0.0) {
    //         std::cout << "closest valid point: " << cv::closestValidPoint< float >(reference_upper_point, depth_frame, 5, 5) << std::endl;
    //     }
    // } catch (std::runtime_error&) {
    //     // cv::drawMarker(hip_img, reference_upper_point, cv::Scalar(0, 255, 0), cv::MARKER_TRIANGLE_DOWN, 10, 2);
    //     cv::imshow("Depth", depth_frame);
    //     cv::imshow("Hip Detection", hip_img);
    //     cv::waitKey(0);
    // }


    // cv::line(hip_img, cv::Point(0, row_idx), cv::Point(hip_img.cols - 1, row_idx), cv::Scalar(0, 155, 155), 2);
    // cv::imshow("Hip Detection", hip_img);
    // cv::imshow("Depth", depth_frame);
    // cv::waitKey(0);
    //--------------------------------------------------------------------------
    /// adjust hip left/right points to match subject hip width
    if (parameters->force_hip_width == true) {
        cv::Point3f hip_left_3D  = cv::imageToWorld< float >(left_hip, depth_frame, *parameters->camera_intrinsics);
        cv::Point3f hip_right_3D = cv::imageToWorld< float >(right_hip, depth_frame, *parameters->camera_intrinsics);
        float hip_width = cv::distance3D(hip_left_3D, hip_right_3D);
        while (hip_width > parameters->reference_hip_width) {
            left_hip.x++;
            right_hip.x--;
            hip_left_3D  = cv::imageToWorld< float >(left_hip, depth_frame, *parameters->camera_intrinsics);
            hip_right_3D = cv::imageToWorld< float >(right_hip, depth_frame, *parameters->camera_intrinsics);
            hip_width = cv::distance3D(hip_left_3D, hip_right_3D);
        }
    }

    //--------------------------------------------------------------------------
    // compute torso center as the midpoint between shoulder and hip midpoints
    // use torso_center_height_factor as the relative value of the distance from hip midpoint to the reference (shoulder/neck)
    cv::Point2i torso_center = cv::Point2i(hip_center.x + parameters->torso_center_height_factor * (reference_upper_point.x - hip_center.x),
                                           hip_center.y + parameters->torso_center_height_factor * (reference_upper_point.y - hip_center.y));

    // cv::drawMarker(hip_img, left_hip, cv::Scalar(255, 0, 0), cv::MARKER_SQUARE, 10, 2);
    // cv::drawMarker(hip_img, right_hip, cv::Scalar(255, 0, 0), cv::MARKER_SQUARE, 10, 2);
    // cv::drawMarker(hip_img, hip_center, cv::Scalar(255, 0, 0), cv::MARKER_SQUARE, 10, 2);
    // cv::imshow("Hip Keypoint Detection", hip_img);
    // cv::imwrite("figs/torso/hip_keypoints_calculation.png", hip_img);
    // cv::waitKey(0);

    return std::vector< cv::Point >{ torso_center, left_hip, hip_center, right_hip };
}


std::vector< cv::Point2i > TorsoDetector::locateLowerKeypoints2(const cv::Mat& depth_frame, const cv::Mat& foreground_mask, const std::vector< cv::Point2i >& upper_keypoints, const TorsoDetectorParameters::Ptr& parameters) {
    // check input foreground mask data type
    if (foreground_mask.type() != CV_8UC1) {
        throw std::invalid_argument("TorsoDetector::locateLowerKeypoints(): invalid input foreground mask");
    }

    //--------------------------------------------------------------------------
    // apply morphological filters to depth foreground
    // eliminates gaps on foreground mask, so that torso pixels are contiguous
    // @note: operations only applied to foreground mask; depth frame may still contain pixels wo/ depth value
    cv::Mat filtered_foreground_mask = foreground_mask;
    // morphological operations
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(parameters->morphological_kernel_size, parameters->morphological_kernel_size));
    // closing (frame is modified in place)
    cv::morphologyEx(filtered_foreground_mask, filtered_foreground_mask, cv::MORPH_CLOSE,  kernel);
    cv::morphologyEx(filtered_foreground_mask, filtered_foreground_mask, cv::MORPH_DILATE, kernel);
    cv::morphologyEx(filtered_foreground_mask, filtered_foreground_mask, cv::MORPH_CLOSE,  kernel);
    cv::morphologyEx(filtered_foreground_mask, filtered_foreground_mask, cv::MORPH_CLOSE,  kernel);

    //--------------------------------------------------------------------------
    // filter input depth image
    // eliminates pontual no-depth pixels and helps extrapolate depth value around shadow areas
    // drawback is that depth reading is not as precise, and introduces an additional computational weight
    cv::Mat filtered_depth_frame = depth_frame;

    //--------------------------------------------------------------------------
    // swep over image from the bottom -> up
    int lowest_torso_row_idx = -1;
    int row_idx = filtered_foreground_mask.rows - 1;
    while (lowest_torso_row_idx == -1 && row_idx > 0) {
        // find lowest row with mask pixels
        uint8_t* foreground_mask_row = filtered_foreground_mask.ptr< uint8_t >(row_idx--);
        for (uint col_idx = 0; col_idx < filtered_foreground_mask.cols; col_idx++) {
            if (foreground_mask_row[col_idx] == 255) {
                // std::cout << foreground_mask.at<uint8_t>(0,filtered_foreground_mask.rows - 1) << std::endl;
                lowest_torso_row_idx = row_idx;
                break;
            }
        }
    }

    //--------------------------------------------------------------------------
    // reconstruct 3D coordinates of reference pixel
    // @todo: use parameters->torso_height with a tolerance value to compensate neck height when parameters->use_neck_reference == true?
    // @note: using shoulder midpoint is kept for legacy, and discouraged; common for head to cause shadow under the chin, leading to null-depth shoulder midpoint
    const cv::Point2i& upper_reference = parameters->use_neck_reference ? upper_keypoints[Keypoint::Neck] : upper_keypoints[Keypoint::ShoulderCenter];

    // check if reference pixel has depth, average local neightbours otherwise
    cv::Point3f upper_reference_3D;
    if (depth_frame.at< float >(upper_reference) == 0) {
        upper_reference_3D = cv::imageToWorld< float >(upper_reference, cv::localAverage< float >(upper_reference, depth_frame), *parameters->camera_intrinsics);
    } else {
        upper_reference_3D = cv::imageToWorld< float >(upper_reference, depth_frame, *parameters->camera_intrinsics);
    }

    //--------------------------------------------------------------------------
    // sweep over image rows down -> up searching for optimal hip row
    // for each row under analysis:
    // 1. find edge pixels (mask == 1 w/ neighbours == 0)
    // 2. compute 2D distances between consecutive edge pixels, find max value and assing left/right hip to both edges
    // 3. compute world distance between hip midpoint and shoulder midpoint
    // 4. repeat 1-3 on row *above* until vertical distance overcomes reference torso height (subject-specific)

    // @note: row_idx is reused/reinitialized
    row_idx = lowest_torso_row_idx;
    // upper limit for hip line
    // implemented for efficiency purposes (avoid needless checking on upper rows -> hip should never be near or above shoulders)
    int min_row_idx = lowest_torso_row_idx - static_cast< int >(parameters->max_hip_height_factor * filtered_foreground_mask.rows);

    // initialize 2D points to be detected (returned) as (0, 0)
    cv::Point2i left_hip(0, 0);
    cv::Point2i right_hip(0, 0);
    cv::Point2i hip_center(0, 0);

    while (row_idx > min_row_idx) {
        // identify edge points (foreground->background transitions and vice-versa) on hip row (a.k.a. pixels with null neighbours)
        std::vector< uint > mask_hip_row_edge_cols;
        uint8_t* foreground_mask_row = filtered_foreground_mask.ptr< uint8_t >(row_idx);
        for (int col_idx = 1; col_idx < filtered_foreground_mask.cols - 1; col_idx++) {
            // if mask point with a non-mask neighbour, append to vector
            if (foreground_mask_row[col_idx] > 0 && (foreground_mask_row[col_idx - 1] == 0 || foreground_mask_row[col_idx + 1] == 0)) {
                mask_hip_row_edge_cols.emplace_back(col_idx);
            }
        }

        // check if enough edges (>= 2) were detected, skip row otherwise
        // under normal circumstances, this should never happen, as foreground extraction will create *at least* two edges on ROI border
        if (mask_hip_row_edge_cols.size() < 2) {
            printf("single edge!\n");
            row_idx--;
            continue;
        }

        // from edge points, calculate horizontal distance in px between consecutive points
        // @note: could be done directly on previous loop without creating a separate edge vector
        //        however code is cleaner and more readable this way; it's a cheap operation anyway with such a small vector.
        int max_width = 0;
        int left_hip_edge_col;
        int right_hip_edge_col;
        for (int idx = 1; idx < mask_hip_row_edge_cols.size(); idx++) {
            int width_px =  mask_hip_row_edge_cols[idx] - mask_hip_row_edge_cols[idx - 1];
            if (width_px > max_width) {
                max_width = width_px;
                left_hip_edge_col  = mask_hip_row_edge_cols[idx-1];
                right_hip_edge_col = mask_hip_row_edge_cols[idx];
            }
        }

        // compute hip midpoint
        left_hip   = cv::Point2i(left_hip_edge_col, row_idx);
        right_hip  = cv::Point2i(right_hip_edge_col, row_idx);
        hip_center = cv::midpoint2D(left_hip, right_hip);

        // update torso measured height (3D distance between hip and shoulder midpoints)
        cv::Point3f hip_center_3D;
        if (depth_frame.at< float >(hip_center) == 0) {
            // @note: using imageToWorld<> overload that takes in single depth value instead of whole frame
            hip_center_3D = cv::imageToWorld< float >(hip_center, cv::localAverage< float >(hip_center, depth_frame), *parameters->camera_intrinsics);
        } else {
            hip_center_3D = cv::imageToWorld< float >(hip_center, depth_frame, *parameters->camera_intrinsics);
        }

        // compute 3D distance between reference neck/shoulder midpoints and estimated hip midpoint
        float torso_height = cv::distance3D(hip_center_3D, upper_reference_3D);

        // check stopping criteria
        // this works because it is assumed at lower row torso height distance between midpoint and reference point is greater than torso heigh!
        // therefore entirly dependent on adequate/appropriate reference values.
        if (torso_height <= parameters->reference_torso_height) {
        // if ((torso_height - parameters->reference_torso_height) <= parameters->torso_height_tolerance) {
            break;
        }

        row_idx--;
    }

    // check if lower point detection was successful, throw exception otherwise!
    if (hip_center.x == 0 && hip_center.y == 0) {
        throw std::runtime_error("TorsoDetector::locateLowerKeypoints(): Unable to locate hip points matching reference torso height; Re-parametrize ");
    }

    //--------------------------------------------------------------------------
    /// adjust hip left/right points to match subject hip width if parametrized to do so.
    if (parameters->force_hip_width == true) {
        cv::Point3f hip_left_3D  = cv::imageToWorld< float >(left_hip, depth_frame, *parameters->camera_intrinsics);
        cv::Point3f hip_right_3D = cv::imageToWorld< float >(right_hip, depth_frame, *parameters->camera_intrinsics);
        float hip_width = cv::distance3D(hip_left_3D, hip_right_3D);
        while ((hip_width > parameters->reference_hip_width || hip_width == 0.0) && left_hip.x < right_hip.x) {
            left_hip.x++;
            right_hip.x--;
            hip_left_3D  = cv::imageToWorld< float >(left_hip, depth_frame, *parameters->camera_intrinsics);
            hip_right_3D = cv::imageToWorld< float >(right_hip, depth_frame, *parameters->camera_intrinsics);
            hip_width = cv::distance3D(hip_left_3D, hip_right_3D);
        }
        // std::cout << hip_width << std::endl;
    }

    //--------------------------------------------------------------------------
    // compute torso center as the midpoint between shoulder and hip midpoints
    // use torso_center_height_factor parameter as the relative value of the distance from hip midpoint to the reference (shoulder/neck)
    cv::Point2i torso_center = cv::Point2i(hip_center.x + parameters->torso_center_height_factor * (upper_reference.x - hip_center.x),
                                           hip_center.y + parameters->torso_center_height_factor * (upper_reference.y - hip_center.y));

    return std::vector< cv::Point2i >{ torso_center, left_hip, hip_center, right_hip };
}


void TorsoDetector::drawKeypoints(cv::Mat& frame, const std::vector< cv::Point2i >& keypoints, bool invert_sides, bool draw_segments, bool draw_aux) {
    cv::Scalar skeleton_point_color(0, 150, 150);
    int        skeleton_point_marker = cv::MARKER_SQUARE;
    cv::Scalar aux_point_color(150, 150, 0);
    int        aux_point_marker      = cv::MARKER_TRIANGLE_DOWN;
    cv::Scalar right_segment_color(255, 0, 0);
    cv::Scalar left_segment_color(0, 0, 255);
    cv::Scalar neutral_segment_color(0, 255, 0);
    //--------------------------------------------------------------------------
    /// [NECK_LEFT, NECK_MID, NECK_RIGHT, SHOULDER_LEFT, SHOULDER_MID, SHOULDER_RIGHT, TORSO_CENTER, HIP_LEFT, HIP_CENTER, HIP_RIGHT]
    cv::drawMarker(frame, keypoints[Keypoint::Neck] /* neck_center */, skeleton_point_color, skeleton_point_marker, 10, 1);
    cv::drawMarker(frame, keypoints[Keypoint::LeftShoulder] /* left shoulder */, skeleton_point_color, skeleton_point_marker, 10, 1);
    cv::drawMarker(frame, keypoints[Keypoint::RightShoulder] /* right_shoulder */, skeleton_point_color, skeleton_point_marker, 10, 1);
    cv::drawMarker(frame, keypoints[Keypoint::TorsoCenter] /* torso center (CoM) */, skeleton_point_color, skeleton_point_marker, 10, 1);
    cv::drawMarker(frame, keypoints[Keypoint::LeftHip] /* left / upper leg */, skeleton_point_color, skeleton_point_marker, 10, 1);
    cv::drawMarker(frame, keypoints[Keypoint::HipCenter] /* hip midpoint (pelvis) */, skeleton_point_color, skeleton_point_marker, 10, 1);
    cv::drawMarker(frame, keypoints[Keypoint::RightHip] /* right hip / upper leg */, skeleton_point_color, skeleton_point_marker, 10, 1);
    /// aux points (no skeleton joint/connection)
    if (draw_aux == true) {
        cv::drawMarker(frame, keypoints[Keypoint::LeftNeckEdge] /* left_neck_edge */, aux_point_color, aux_point_marker, 5, 1);
        cv::drawMarker(frame, keypoints[Keypoint::RightNeckEdge] /* right_neck_edge */, aux_point_color, aux_point_marker, 5, 1);
        cv::drawMarker(frame, keypoints[Keypoint::ShoulderCenter] /* shoulder midpoint (chest) */, aux_point_color, aux_point_marker, 5, 1);
    }
    if (draw_segments == true) {
        /// draw line segments
        cv::line(frame, keypoints[Keypoint::Neck],        keypoints[Keypoint::TorsoCenter], neutral_segment_color, 2);  // neck -> torso midpoint (upper spine)
        cv::line(frame, keypoints[Keypoint::Neck],        keypoints[Keypoint::LeftShoulder], left_segment_color, 2);     // neck -> left shoulder
        cv::line(frame, keypoints[Keypoint::Neck],        keypoints[Keypoint::RightShoulder], right_segment_color, 2);    // neck -> right shoulder
        cv::line(frame, keypoints[Keypoint::TorsoCenter], keypoints[Keypoint::HipCenter], neutral_segment_color, 2);  // torso midpoint -> pelvis (lower spine)
        cv::line(frame, keypoints[Keypoint::HipCenter],   keypoints[Keypoint::LeftHip], left_segment_color, 2);     // pelvis -> left hip
        cv::line(frame, keypoints[Keypoint::HipCenter],   keypoints[Keypoint::RightHip], right_segment_color, 2);    // pelvis -> right hip
    }
}


float TorsoDetector::detectionQuality(const cv::Mat& depth_frame, const std::vector< cv::Point2i >& keypoints, const TorsoDetectorParameters::Ptr& parameters) {
    // ...
    return 0.0;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace asbgo::vision
