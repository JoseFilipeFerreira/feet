#ifndef INCLUDE_CVUTILS_HPP
#define INCLUDE_CVUTILS_HPP

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#include <vector>
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>  // cv::imshow, cv:: waitKey

//------------------------------------------------------------------------------
/// @brief      Generic public types and templated/free functions extending and/or simplifying usage of OpenCV library, declared within cv:: namespace
///
namespace cv {
//------------------------------------------------------------------------------
/// @brief      Generic value ranges
///
/// @note       Harcoded value ranges for different value types/depth (e.g. 8/16/32 uint images)
///
typedef const float* ColorRange;
static constexpr float RANGE_POLAR[] = {0, 179};
static constexpr float RANGE_8[]     = {0, 255};
static constexpr float RANGE_16[]    = {0, 65536};
static constexpr float RANGE_32[]    = {0, 4294967296};

//------------------------------------------------------------------------------
/// @brief      Color space ranges
///
/// @note       Useful when building color histograms, as OpenCV's functions need multiple (channels) value ranges to be specified
///
template < int MIN_C0, int MAX_C0, int MIN_C1, int MAX_C1, int MIN_C2, int MAX_C2 >
struct ColorModelInfo {
    const int min_c0 = MIN_C0;
    const int max_c0 = MAX_C0;
    const int min_c1 = MIN_C1;
    const int max_c1 = MAX_C1;
    const int min_c2 = MIN_C2;
    const int max_c2 = MAX_C2;
};
// template specialization
typedef ColorModelInfo< 0, 179,   0, 255,   0, 255   > HSVRange_8;
typedef ColorModelInfo< 0, 255,   0, 255,   0, 255   > BGRRange_8;
typedef ColorModelInfo< 0, 65536, 0, 65536, 0, 65536 > BGRRange_16;

//------------------------------------------------------------------------------
/// @brief      Time stamp type
///
/// @note       Required/used by different classes, single declaration in external header allows fast/practical way to change type if necessary
///
using TimeStamp = long double;

//------------------------------------------------------------------------------
/// @brief      Templated 2D shape as a group of 3D points
///             Useful to represent shape contours
///             Covenience public type alias for code readibility
///
template < typename T >
using Shape2 = typename std::vector< Point_< T > >;
typedef Shape2< int >    Shape2i;
typedef Shape2< float >  Shape2f;
typedef Shape2< double > Shape2d;

//------------------------------------------------------------------------------
/// @brief      Templated 3D shape as a group of 3D points
///             Covenience public type alias for code readibility
///
template < typename T >
using Shape3 = typename std::vector< Point3_< T > >;
typedef Shape3< int >    Shape3i;
typedef Shape3< float >  Shape3f;
typedef Shape3< double > Shape3d;

//------------------------------------------------------------------------------
/// @brief      Computes the centroid from a set of 2D points
///
/// @param[in]  points  Point vector
///
/// @tparam     DT      Point data type
///
/// @return     Point_< DT > instance (same as input type)
///
template < typename DT >
Point_< float > centroid2D(const Shape2< DT >& points) {
    Point_< float > centroid(0, 0);
    for (auto& point : points) {
        centroid.x += (1.0 / points.size()) * point.x;
        centroid.y += (1.0 / points.size()) * point.y;
    }
    return centroid;
}


//------------------------------------------------------------------------------
/// @brief      Computes the centroid of a cv::Rect object (2D rectangle)
///
/// @param[in]  rectangle  The rectangle
///
/// @tparam     DT         Point data type
///
/// @return     Point_< DT > instance (same as input type)
///
template < typename DT >
Point_< float > centroid2D(const Rect_< DT > rectangle) {
    Point_< float > centroid(0, 0);
    centroid.x = rectangle.x + 0.5 * rectangle.width;
    centroid.y = rectangle.y + 0.5 * rectangle.height;
    return centroid;
}


//------------------------------------------------------------------------------
/// @brief      Computes the centroid of a 2D contour.
///             Differs from centroid2D as it accepts a closed 2D shape and also uses inner points.
///             Useful to compute 'CoM' of a 2D shape/contour.
///
/// @param[in]  contour    Vector of contour points (2D)
/// @param[in]  reference  Reference frame/size
///
/// @tparam     DT         Contour point data type
///
/// @return     2D point (float) with estimated centroid coordinates.
///
template < typename DT >
Point_< float > contourCentroid(const Shape2< DT > contour, const Mat& reference = Mat()) {
    Point_< float > centroid(0, 0);
    // loop over contour points to find max and min
    Point2f min(-1.0, -1.0);
    Point2f max(-1.0, -1.0);
    for (auto& point : contour) {
        if (point.x < min.x || min.x < 0.0) {
            min.x = point.x;
        }
        if (point.y < min.y || min.y < 0.0) {
            min.y = point.y;
        }
        if (point.x > max.x || max.x < 0.0) {
            max.x = point.x;
        }
        if (point.y > max.y || max.y < 0.0) {
            max.y = point.y;
        }
    }
    // check if reference is empty (default parameter if not provided), ignore max point otherwise.
    // allows getting in-frame centroid if contour is partially out of the image
    if (reference.empty() == false) {
        max.x = reference.cols;
        max.y = reference.rows;
    }
    // average coordiantes of points *within* countour shape!
    int pixel_count = 0;
    for (int row_idx = min.y; row_idx < max.y; row_idx++) {
        for (int col_idx = min.x; col_idx < max.x; col_idx++) {
            Point2i pixel(col_idx, row_idx);
            if (pointPolygonTest(contour, pixel, false) >= 0.0) {
                centroid.x += pixel.x;
                centroid.y += pixel.y;
                pixel_count++;
            }
        }
    }
    centroid.x *= (1.0 / pixel_count);
    centroid.y *= (1.0 / pixel_count);
    return centroid;
}


//------------------------------------------------------------------------------
/// @brief      Computes the centroid of a set of 3D points
///
/// @param[in]  points  Point vector
///
/// @tparam     DT      Point data type
///
/// @return     Point_< DT > instance (same as input type)
///
template < typename DT >
Point3_< float > centroid3D(const Shape3< DT >& points) {
    Point3_< float > centroid(0, 0, 0);
    for (auto& point : points) {
        centroid.x += (1.0 / points.size()) * point.x;
        centroid.y += (1.0 / points.size()) * point.y;
        centroid.z += (1.0 / points.size()) * point.z;
    }
    return centroid;
}

//------------------------------------------------------------------------------
/// @brief      Initializes a 3D point from 2D coordinates
///             Provided for code readibility, as Point3_< T > class does not provide a copy constructor from 2D point
///
/// @param[in]  point  Input 2D point
///
/// @tparam     DT_0   Output data type
/// @tparam     DT_1   Input data type
///
/// @return     Point3_< DT_0 > instance
///
template < typename DT_0, typename DT_1 >
Point_< DT_0 > to2D(const Point3_< DT_1 >& point) {
    return Point_< DT_0 > (point.x, point.y);
}

//------------------------------------------------------------------------------
/// @brief      Initializes a 3D point from 2D coordinates
///             Provided for code readibility, as Point3_< T > class does not provide a copy constructor from 2D point
///
/// @param[in]  point  Input 2D point
///
/// @tparam     DT_0   Output data type
/// @tparam     DT_1   Input data type
///
/// @return     Point3_< DT_0 > instance
///
template < typename DT_0, typename DT_1 >
Point3_< DT_0 > to3D(const Point_< DT_1 >& point) {
    return Point3_< DT_0 > (point.x, point.y, 0.0);
}

//------------------------------------------------------------------------------
/// @brief      Euclidean distance between two different 2D points
///             Implementd as the non-negative square root of squared coordinate differences
///
/// @param[in]  point        2D point #1
/// @param[in]  other_point  2D point #2
///
/// @tparam     DT_0         Input data type #1
/// @tparam     DT_1         Input data type #2
///
/// @return     floating point value of the absolute (straight line) distance between the two points
///
template < typename DT_0, typename DT_1 >
float distance2D(const Point_< DT_0 >& point, const Point_< DT_1 >& other_point) {
    return sqrt(pow(other_point.x - point.x, 2) + pow(other_point.y - point.y, 2));
}

//------------------------------------------------------------------------------
/// @brief      Euclidean distance between two different 3D points
///             Implementd as the non-negative square root of squared coordinate differences
///
/// @param[in]  point        3D point #1
/// @param[in]  other_point  3D point #2
///
/// @tparam     DT_0         Input data type #1
/// @tparam     DT_1         Input data type #2
///
/// @return     floating point value of the absolute (straight line) distance between the two points
///
template < typename DT_0, typename DT_1 >
float distance3D(const Point3_< DT_0 >& point, const Point3_< DT_1 >& other_point) {
    return sqrt(pow(other_point.x - point.x, 2) + pow(other_point.y - point.y, 2) + pow(other_point.z - point.z, 2));
}

//------------------------------------------------------------------------------
/// @brief      Midpoint of straight line between two 2D points
///
/// @param[in]  point        2D point #1
/// @param[in]  other_point  2D point #2
///
/// @tparam     DT_0         Input data type #1
/// @tparam     DT_1         Input data type #2
///
/// @return     Cordinates of 2D point between two input points
///
template < typename DT_0, typename DT_1 >
Point_< float > midpoint2D(const Point_< DT_0 >& point, const Point_< DT_1 >& other_point) {
    return Point_< float > (point.x + 0.5 * (other_point.x - point.x), point.y + 0.5 * (other_point.y - point.y));
}

//------------------------------------------------------------------------------
/// @brief      Midpoint of straight line between two 3D points
///
/// @param[in]  point        3D point #1
/// @param[in]  other_point  3D point #2
///
/// @tparam     DT_0         Input data type #1
/// @tparam     DT_1         Input data type #2
///
/// @return     Cordinates of 3D point between two input points
template < typename DT_0, typename DT_1 >
Point3_< float > midpoint3D(const Point3_< DT_0 >& point, const Point3_< DT_1 >& other_point) {
    return Point3_< float > (point.x + 0.5 * (other_point.x - point.x), point.y + 0.5 * (other_point.y - point.y), point.z + 0.5 * (other_point.z - point.z));
}

//------------------------------------------------------------------------------
/// @brief      Multiplication operator between a 2D point and a matrix
///             Useful to apply 2D transformations to points
///
/// @param[in]  point   Input 2D point
/// @param[in]  matrix  Input matrix
///
/// @tparam     iT      input data type
///
/// @return     The result of the multiplication
///
template < typename iT >
Point_< iT > operator*(const Point_< iT >& point, const Mat& matrix) {
    if (matrix.rows != 2) {
        throw std::invalid_argument("cv::operator*(): invalid input matrix (rows != 2)");
    }
    Mat point_mat(point, false /* to avoid memory copy*/);
    Mat result = point_mat.t() * matrix;
    return Point_< iT >(result.at< iT > (0, 0), result.at< iT >(0, 1));
}

//------------------------------------------------------------------------------
/// @brief      Multiplication operator between a 3D point and a matrix
///             Useful to apply 3D transformations to points
///
/// @param[in]  point   Input 3D point
/// @param[in]  matrix  Input matrix
///
/// @tparam     iT      input data type
///
/// @return     The result of the multiplication
///
template < typename iT >
Point3_< iT > operator*(const Point3_< iT >& point, const Mat& matrix) {
    if (matrix.rows != 3) {
        std::cout << matrix.rows << "x" << matrix.cols << std::endl;
        throw std::invalid_argument("cv::operator*(): invalid input matrix (rows != 3)");
    }
    // @note: creating a Mat from a point rturns a 3x1 matrix (though printing Mat::size() returns the othr way around)
    Mat point_mat(point, false /* to avoid memory copy*/);
    Mat result = point_mat.t() * matrix;
    // @note: result should be of size 1x3 max
    return Point3_< iT >(result.at< iT > (0, 0), result.at< iT >(0, 1), result.at< iT >(0, 2));
}

//------------------------------------------------------------------------------
/// @brief      Applies a generic 3D transformation to a point in-place (no copy)
///             Differs from * operator in it implements reverse operation (mat * point) and requires 4x4 matrix
///
/// @param      point             Input 3D point
/// @param[in]  transform_matrix  3D transformation matrix (rotation |
///                               translation) of size 4x4
///
/// @tparam     T                 Input data type
///
template < typename T >
void transform3D(Point3_< T >* point, const Mat& transform_matrix) {
    Mat point_mat(*point);
    if (transform_matrix.cols == 4) {
        point_mat.push_back< T >(1.0);
    }
    Mat new_point_mat = transform_matrix * point_mat;
    point->x = new_point_mat.at< T >(0);
    point->y = new_point_mat.at< T >(1);
    point->z = new_point_mat.at< T >(2);
}

//------------------------------------------------------------------------------
/// @brief      Applies a generic 3D transformation to a point
///             Differs from * operator in it implements reverse operation (mat * point) and requires 4x4 matrix
///             Additional overload with templated return type
///
/// @param      point             Input 3D point
/// @param[in]  transform_matrix  3D transformation matrix (rotation |
///                               translation) of size 4x4
///
/// @tparam     oT                Output data type
/// @tparam     iT                Input data type
///
/// @return     new 3D point instance with transformation applied
///
template < typename oT, typename iT >
Point3_< oT > transform3D(const Point3_< iT >& point, const Mat& transform_matrix) {
    Point3_< oT > transformed = point;
    transform3D(&transformed, transform_matrix);
    return transformed;
}


inline Mat inverseTransformMat(const Mat& transform_matrix) {
    Mat rotation_inv = transform_matrix(cv::Rect(0, 0, 3, 3)).clone().inv();
    Mat translation_inv = -1 * rotation_inv * transform_matrix(cv::Rect(3, 0, 1, 3)).clone();
    // construct output matrix
    Mat transform_inv;
    hconcat(rotation_inv, translation_inv, transform_inv);
    // vconcat(transform_inv, transform_matrix(cv::Rect(0, 3, 4, 1), transform_inv);
    transform_inv.push_back(transform_matrix(cv::Rect(0, 3, 4, 1)));
    // std::cout << "Transform ^-1: \n" << transform_inv << std::endl;
    return transform_inv;
}

inline Mat rotationMatXX(double angle) {
    Mat rotation = Mat::eye(4, 4, CV_64FC1);
    rotation.at< double >(1, 1) = cos(angle);
    rotation.at< double >(1, 2) = -1.0 * sin(angle);
    rotation.at< double >(2, 1) = sin(angle);
    rotation.at< double >(2, 2) = cos(angle);
    return rotation;
}

inline Mat rotationMatYY(double angle) {
    Mat rotation = Mat::eye(4, 4, CV_64FC1);
    rotation.at< double >(0, 0) = cos(angle);
    rotation.at< double >(0, 2) = sin(angle);
    rotation.at< double >(2, 0) = -1.0 * sin(angle);
    rotation.at< double >(2, 2) = cos(angle);
    return rotation;
}

inline Mat rotationMatZZ(double angle) {
    Mat rotation = Mat::eye(4, 4, CV_64FC1);
    rotation.at< double >(0, 0) = cos(angle);
    rotation.at< double >(0, 1) = -1.0 * sin(angle);
    rotation.at< double >(1, 0) = sin(angle);
    rotation.at< double >(1, 1) = cos(angle);
    return rotation;
}

template < typename T >
void inverseTransform3D(Point3_< T >* point, const Mat& transform_matrix, bool factorize = false) {
    Mat point_mat(*point);
    point_mat.push_back< T >(1.0);
    Mat new_point_mat;
    if (factorize == true) {
        // get inverse translation
        Mat translation_inv = Mat::eye(4, 4, transform_matrix.type());
        translation_inv.at< double >(0, 3) = -1.0 * transform_matrix.at< double >(0, 3);  // -t_x
        translation_inv.at< double >(1, 3) = -1.0 * transform_matrix.at< double >(1, 3);  // -t_y
        translation_inv.at< double >(2, 3) = -1.0 * transform_matrix.at< double >(2, 3);  // -t_z
        // std::cout << translation_inv << std::endl;
        // get inverse rotation
        Mat rotation_inv = Mat::eye(4, 4, transform_matrix.type());
        rotation_inv.at< double >(0, 0) = transform_matrix.at< double >(0, 0);
        rotation_inv.at< double >(0, 1) = transform_matrix.at< double >(0, 1);
        rotation_inv.at< double >(0, 2) = transform_matrix.at< double >(0, 2);
        rotation_inv.at< double >(1, 0) = transform_matrix.at< double >(1, 0);
        rotation_inv.at< double >(1, 1) = transform_matrix.at< double >(1, 1);
        rotation_inv.at< double >(1, 2) = transform_matrix.at< double >(1, 2);
        rotation_inv.at< double >(2, 0) = transform_matrix.at< double >(2, 0);
        rotation_inv.at< double >(2, 1) = transform_matrix.at< double >(2, 1);
        rotation_inv.at< double >(2, 2) = transform_matrix.at< double >(2, 2);
        rotation_inv = rotation_inv.t();
        resize(rotation_inv, rotation_inv, Size(4, 4));
        // apply transformations in order
        new_point_mat = translation_inv * point_mat;
        new_point_mat = rotation_inv * new_point_mat;
    } else {
        new_point_mat = transform_matrix.inv() * point_mat;
    }
    // rotation_inv.at< double >(3, 3) = 1.0;
    // std::cout << rotation_inv << std::endl;
    // Mat new_point_mat = translation_inv * point_mat;
    // Mat new_point_mat = transform_matrix.inv() * point_mat;
    // @note  3D transformations (matrix multiplications) are not commutative!
    // apply inverse translation THEN inverse rotation
    // new_point_mat = rotation_inv * new_point_mat;
    point->x = new_point_mat.at< T >(0);
    point->y = new_point_mat.at< T >(1);
    point->z = new_point_mat.at< T >(2);
}


//------------------------------------------------------------------------------
/// @brief      Converts pixel coordinates to 2D point (inverts 'x' and 'y' coordinates)
///             To be applied when pixel coordinates are being used in a Point2i instance
///
/// @param[in]  pixel  2D point where 'x' = row and 'y' = col
///`
/// @return     new 2D point with swapped coordinates
///
inline Point2i pixelToPoint(const Point2i& pixel) {
    return Point2i(pixel.y, pixel.x);
}

//------------------------------------------------------------------------------
/// @brief      Converts multiple pixel coordinates to 2D point (inverts 'x' and 'y' coordinates)
///             To be applied when pixel coordinates are being used in a Point2i instance
///
/// @param[in]  pixels  Vector of 2D point where 'x' = row and 'y' = col
///
/// @return     new 2D point vector with swapped coordinates
///
inline std::vector< Point2i > pixelToPoint(const std::vector< Point2i >& pixels) {
    std::vector< Point2i > points;
    for (const auto& pixel : pixels) {
        points.emplace_back(pixelToPoint(pixel));
    }
    return points;
}

//------------------------------------------------------------------------------
/// @brief      finds nearest non-null point
///
/// @note       useful to escape random empty pixels in depth images
///
template < typename T >
inline Point2i closestValidPoint(const Point2i& point, const cv::Mat& frame, int range_x = 2, int range_y = 2) {
    if (frame.at< T >(point) > 0) {
        return point;
    }
    for (int x = point.x - range_x; x <= point.x + range_x; x++) {
        for (int y = point.y - range_y; y <= point.y + range_y; y++) {
            Point2i nb(x, y);
            if (nb.x >= 0 && nb.x < frame.cols && nb.y >= 0 && nb.y < frame.rows) {
                if (frame.at< T >(nb) > 0) {
                    return nb;
                }
            }
        }
    }
    throw std::runtime_error("closestValidPoint(): no valid neighbour point!");
}


//------------------------------------------------------------------------------
/// @brief      averages
///
/// @note       useful to escape random empty pixels in depth images
///
/// @param[in]  point    The point
/// @param[in]  frame    The frame
/// @param[in]  range_x  The range x
/// @param[in]  range_y  The range y
///
/// @tparam     T        { description }
///
/// @return     { description_of_the_return_value }
///
template < typename T >
inline T localAverage(const Point2i& point, const cv::Mat& frame, int range_x = 5, int range_y = 5) {
    if (frame.channels() > 1) {
        throw std::invalid_argument("cv::localAverage(): invalid input frame!");
    }
    float sum = 0.0;
    int count = 0;
    for (int x = point.x - range_x; x <= point.x + range_x; x++) {
        for (int y = point.y - range_y; y <= point.y + range_y; y++) {
            Point2i nb(x, y);
            if (nb.x >= 0 && nb.x < frame.cols && nb.y >= 0 && nb.y < frame.rows) {
                float val = frame.at< T >(nb);
                if (val > 0) {
                    sum += val;
                    count++;
                }
            }
        }
    }
    return (sum / count);
}


//------------------------------------------------------------------------------
/// @brief      Converts 2D image coordinates to 3D world cordinates (3D reconstruction)
///             Applies camera's intrinsic transformation to input pixels
///
/// @param[in]  point             Input 2D coordinates (col, row)
/// @param[in]  depth_frame       Input depth frame (CV_32FC1 type) with depth values (m)
/// @param[in]  intrinsic_matrix  Camera's intrinsic parametrization (3x3)
/// @param[in]  invert_x          Axis invert flag for 'x' coordinate
/// @param[in]  invert_y          Axis invert flag for 'y' coordinate
///
/// @tparam     T                 Output point data type
///
/// @return     3D point instance (Point3_< T >)
///
template < typename T >
inline Point3_< T > imageToWorld(const Point2i& point, float depth, const Mat& intrinsic_matrix, bool invert_x = false, bool invert_y = false) {
    if (depth == 0.0) {
        // depth should not be null, implies a non-valid point
        // using a high value (> max depth theshold, for points far away) leads to weird 'x' and 'y' values
        // using the origin facilitates checking if a point is valid or not, because points will (*should*) never be @(0, 0, 0)
        return Point3_< T >(0.0, 0.0, 0.0);
    }
    // camera params
    double c_x = intrinsic_matrix.at< double >(0, 2);
    double c_y = intrinsic_matrix.at< double >(1, 2);
    double f_x = intrinsic_matrix.at< double >(0, 0);
    double f_y = intrinsic_matrix.at< double >(1, 1);
    // calculate individual transformations
    Point3d point_3D;
    point_3D.x = ((point.x - c_x) * depth) / f_x;
    point_3D.y = ((point.y - c_y) * depth) / f_y;
    point_3D.z = depth;
    //
    if (invert_x == true) {
        point_3D.x *= -1;
    }
    if (invert_y == true) {
        point_3D.y *= -1;
    }
    return point_3D;
}

template < typename T >
inline Point3_< T > imageToWorld(const Point2i& point, const Mat& depth_frame, const Mat& intrinsic_matrix, bool invert_x = false, bool invert_y = false, bool average_if_null = false) {
    // float depth = depth_frame.at< float >(point);
    // if (average_if_null == true && depth == 0) {
    //     depth = localAverage< float >(point, depth_frame);
    // }
    // @note: assumed that depth frame data type is float, may need to check
    return imageToWorld< T >(point, depth_frame.at< float >(point), intrinsic_matrix, invert_x, invert_y);
}


//------------------------------------------------------------------------------
/// @brief      Projects 3D world coordinates to 2D image reference frame
///
/// @param[in]  point_3d          Input 3D coordina tes
/// @param[in]  intrinsic_matrix  Camera's intrinsic matrix
/// @param[in]  invert_x          Axis invert flag for 'x' coordinate
/// @param[in]  invert_y          Axis invert flag for 'y' coordinate
///
/// @tparam     oT                Output coordinate data type
/// @tparam     iT                Input coordinate data type
///
/// @return     2D point instance (Point_< T >)
///
template < typename oT, typename iT >
inline Point_< oT > worldToImage(const Point3_< iT >& point_3d, const Mat& intrinsic_matrix, bool invert_x = false, bool invert_y = false) {
    // invert axis if requested
    int i_x = invert_x ? -1 : 1;
    int i_y = invert_y ? -1 : 1;
    // camera params
    double c_x = intrinsic_matrix.at< double >(0, 2);
    double c_y = intrinsic_matrix.at< double >(1, 2);
    double f_x = intrinsic_matrix.at< double >(0, 0);
    double f_y = intrinsic_matrix.at< double >(1, 1);
    if (point_3d.z == 0) {
        return Point_< oT >((i_x * point_3d.x * f_x) + c_x, (i_y * point_3d.y * f_y) + c_y);
    }
    // calculate individual transformations
    Point_< oT > point_2d;
    point_2d.x = ((i_x * point_3d.x * f_x) / point_3d.z) + c_x;
    point_2d.y = ((i_y * point_3d.y * f_y) / point_3d.z) + c_y;

    return point_2d;
}

//------------------------------------------------------------------------------
/// @brief      Estimates vertical pitch angle from a depth image, using two reference points
///
/// @note       Assumes horizontal alignment (no lateral tilt) and both reference points to be vertically aligned
///
/// @param[in]  depth_frame       Input depth frame, CV_32FC1 type
/// @param[in]  point             Input reference point #1
/// @param[in]  other_point       Input reference point #2
/// @param[in]  intrinsic_matrix  Camera's intrinsic matrix
///
/// @return     Floating point value of vertical pitch angle
///
inline float estimateVerticalPitch(const Mat& depth_frame, Point2i point, Point2i other_point, const Mat& intrinsic_matrix) {
    if (point.x != other_point.x) {
        throw std::invalid_argument("cv::estimateVerticalPitch(): points not vertically aligned");
    }
    if (depth_frame.channels() > 1) {
        throw std::invalid_argument("cv::estimateVerticalPitch(): invalid input frame (channels > 1)");
    }
    // project points into world 3D coordinates
    // @note: points should be expressed as (col, row)
    Point3f point_3D       = imageToWorld< float >(point, depth_frame, intrinsic_matrix);
    Point3f other_point_3D = imageToWorld< float >(other_point, depth_frame, intrinsic_matrix);
    // std::cout << point.x     << ", " << other_point.x    << std::endl;
    // std::cout << point_3D  << ", " << other_point_3D << std::endl;
    float dz = abs(point_3D.z - other_point_3D.z);
    float dy = abs(point_3D.y - other_point_3D.y);
    return atan(dz / dy);
    // convert to degree value
}






//------------------------------------------------------------------------------
/// @brief      Estimates vertical and horizontal orientation of a surface within the camera's FoV.
///
/// @param[in]  depth_frame       Depth frame
/// @param[in]  points            Surface points i.e. top-left and bottom right vertices of a ROI w/ pixels corresponding to the surface.
/// @param[in]  intrinsic_matrix  Camera intrisics for 3D reconstruction.
///
/// @return     Instance of cv::Point2f with horizontal ('x') and vertical ('y') angles (rad).
/// 
typedef cv::Point2f SurfaceOrientation;
inline SurfaceOrientation estimateSurfaceOrientation2D(const Mat& depth_frame, const std::vector< cv::Point2i >& surface_points, const Mat& intrinsic_matrix) {
    if (surface_points.size() < 2) {
        throw std::invalid_argument("cv::estimateSurfaceOrientation2D(): insufficient input points");
    }
    if (surface_points[0].x <= surface_points[1].x || surface_points[0].y <= surface_points[1].y) {
        throw std::invalid_argument("cv::estimateSurfaceOrientation2D(): input points not properly aligned");
    }
    // project points into world 3D coordinates
    // extrapolates additional points assuming 
    cv::Point3f point_3D       = cv::imageToWorld< float >(surface_points[0], depth_frame, intrinsic_matrix);
    cv::Point3f point_3D_side  = cv::imageToWorld< float >(cv::Point(surface_points[1].x, surface_points[0].y), depth_frame, intrinsic_matrix);
    cv::Point3f point_3D_below = cv::imageToWorld< float >(cv::Point(surface_points[0].x, surface_points[1].y), depth_frame, intrinsic_matrix);
    if (point_3D.z == 0.0 || point_3D_side.z == 0.0 || point_3D_below.z  == 0.0) {
        throw std::invalid_argument("cv::estimateSurfaceOrientation2D() : invalid input points (depth == 0.0)");
    }
    // compute horizontal and vertical distances
    float dx       = abs(point_3D.x - point_3D_side.x);
    float dy       = abs(point_3D.y - point_3D_below.y);
    float dz_side  = abs(point_3D.z - point_3D_side.z);
    float dz_below = abs(point_3D.z - point_3D_below.z);
    // populate output object
    cv::Point2f surface_orientation;
    surface_orientation.x = atan(dz_side  / dx);
    surface_orientation.y = atan(dz_below / dy);
    return surface_orientation;
}

//------------------------------------------------------------------------------
/// @brief      Finds the largest shapes (greater area) in a 2D shape (e.g. contour) container
///
/// @param[in]  shapes    Vector of 2D shapes (vector of 2D points)
/// @param[in]  n         Number of largest shapes to return
/// @param[in]  max_area  Maximum area threshold
/// @param[in]  min_area  Minimum area threshold
///
/// @tparam     T         Input shape coordinate data type
///
/// @return     Vector of indexes of largest shapes on input container, sorted by largest to smallest
///
template < typename T >
inline std::vector< int > largestShapeIndex(const std::vector< Shape2< T > > shapes, uint n = 1, float max_area = 100000, float min_area = 0) {
    std::vector< float > max_areas   (n, 0.0);
    std::vector< int >   max_indexes (n, -1);
    // loop over shapes
    for (uint shape_idx = 0; shape_idx < shapes.size(); shape_idx++) {
        float area = cv::contourArea(shapes[shape_idx]);
        // check if minimum area requirement is verified
        if (area < min_area || area < max_areas.back() || area > max_area) {
            continue;
        }
        // loop over max list
        // compare with saved area values
        for (uint max_idx = 0; max_idx < n; max_idx++) {
            if (area > max_areas[max_idx]) {
                // push max list one position down
                for (int idx = max_idx + 1; idx < n; idx++) {
                    max_areas[idx]   = max_areas[idx - 1];
                    max_indexes[idx] = max_indexes[idx - 1];
                }
                // assign new ith max on list
                max_areas[max_idx]   = area;
                max_indexes[max_idx] = shape_idx;
                break;
            }
        }
    }
    return max_indexes;
}

//------------------------------------------------------------------------------
/// @brief      Draws convexity defects from a contour and its convex hull
///
/// @param      image      Input image
/// @param[in]  contour    Contour (Non-convex)
/// @param[in]  defects    Convexity defects (as Vec4i, cf. cv::convexityDefects() definition)
/// @param[in]  color      Line color
/// @param[in]  thickness  Line thickness
/// @param[in]  marker     Draw marker flag, to enable drawing of start/stop/farthest contour points
///
inline void drawConvexityDefects(Mat* image, const Shape2i& contour, const std::vector< cv::Vec4i >& defects, const Scalar& color = Scalar::all(255), int thickness = 1, bool marker = false) {
    // use idx to draw only a single defect!
    // loop over each defect
    for (const auto& defect : defects) {
        // drawContours(*image, std::v);
        uint start_idx = defect[0];
        uint end_idx   = defect[1];
        uint far_idx   = defect[2];
        uint distance  = defect[3] / 256.0;
        // loop over defect contour points
        line(*image, contour[start_idx], contour[end_idx], color, thickness);
        for (int idx = start_idx; idx < end_idx; idx++) {
            line(*image, contour[idx], contour[idx + 1], color, thickness);
        }
        // draw marker on the farthest point from convex hull (concavity)
        if (marker == true) {
            drawMarker(*image, contour[far_idx], cv::Scalar(255, 0, 0), cv::MARKER_TILTED_CROSS, 10, thickness);
        }
    }
}
//------------------------------------------------------------------------------
/// @brief      Finds polygon element (e.g. contour point) closest to a 2D point
///
/// @param[in]  point    Input 2D point
/// @param[in]  polygon  Input 2D polygon (cv::Shape2i)
///
/// @tparam     T        Input point coordinate data type
///
/// @return     Index of closest point in input shape container
///
template < typename T >
inline uint closestPolygonPoint(const Point_<T>& point, const Shape2i& polygon) {
    if (polygon.size() == 1) {
        return 0;
    }
    uint min_idx = 0;
    float min_distance = cv::distance2D(point, polygon[0]);
    for (int idx = 0; idx < polygon.size(); idx++)  {
        float distance = cv::distance2D(point, polygon[idx]);
        if (distance < min_distance) {
            min_distance = distance;
            min_idx = idx;
        }
    }
    return min_idx;
}

//------------------------------------------------------------------------------
/// @brief      Finds convex polygon element (e.g. contour point) closest to a 2D point
///             Differs from closestPolygonPoint in that input polygon must be convex, otherwise local minima will be detected as closest points
///
/// @param[in]  point    Input 2D point
/// @param[in]  polygon  Input 2D polygon
/// @param[in]  check    Check flag to force convexity check on input polygon
///
/// @tparam     T        { description }
///
/// @throws     std::invalid_argument if input polygon is not convex
///
/// @return     { description_of_the_return_value }
///
template < typename T >
inline uint closestConvexPolygonPoint(const Point_<T>& point, const Shape2i& polygon, bool check = false) {
    if (polygon.size() == 1) {
        return 0;
    }
    if (check == true){
        if (isContourConvex(polygon) == false) {
            throw std::invalid_argument("...");
        }
    }
    // // only works for convex shapes! otherwise gets stuck in local minima
    uint idx = static_cast< uint >(polygon.size() * 0.5);
    while (cv::distance2D(point, polygon[idx + 1]) <= cv::distance2D(point, polygon[idx])) {
        printf("incrementing\n");
        idx++;
        if (idx == polygon.size()) {
            idx = 0;
        }
    }
    while (cv::distance2D(point, polygon[idx - 1]) <= cv::distance2D(point, polygon[idx])) {
        printf("decrementing\n");
        idx--;
        if (idx < 0) {
            idx = polygon.size() - 1;
        }
    }
    return idx;
}

//------------------------------------------------------------------------------
/// @brief      Extracts a contour subset
///
/// @note       Useful to extract the convex hull when the indexes are already known, avoiding recomputation of convex hull
///
/// @param[in]  contour  Input contour (Shape2i)
/// @param[in]  indexes  Point indexes for subset contour points
///
/// @return     New 2D Shape2i instance populated with subcontour points
///
inline Shape2i contourSubset(const Shape2i& contour, const std::vector< int >& indexes) {
    Shape2i points(indexes.size());
    for (int idx = 0; idx < points.size(); idx++) {
        // if (indexes[idx] >= contour.size()) {
        //     throw std::invalid_argument("cv::contourSubset(): invalid input index");
        // }
        points[idx] = contour[indexes[idx]];
    }
    return points;
}

//------------------------------------------------------------------------------
/// @brief      Extracts a contour subset from point sequence information
///
/// @param[in]  contour           Input contour (with sorted points)
/// @param[in]  convexity_defect  Convexity defect (Vec4i), with [start_idx, end_idx, farthest_idx]
///
/// @return     New 2D Shape2i instance populated with subcontour points
///
inline Shape2i contourSubset(const Shape2i& contour, const Vec4i& convexity_defect) {
    /// @note: end point index may be lower than start index, when defect includes start/end of contour
    Shape2i points;
    points.reserve(abs(convexity_defect[1] - convexity_defect[0]) + 1);
    for (int idx = convexity_defect[0]; idx != convexity_defect[1] + 1; idx++) {
        // reset index back to start if reached end of array
        if (idx == contour.size()) {
            idx = 0;
        }
        points.emplace_back(contour[idx]);
    }
    return points;
}

//------------------------------------------------------------------------------
/// @brief      Truncates value according to input range object
///
/// @note       Differs from cv::saturate_cast as range is passed as function argument (runtime) instead of as template parameter
///             Additionally, allows differnt input/output types, implicitely casting between types if necessary
///
/// @param[in]  value  Input value
/// @param[in]  range  Input value range
///
/// @tparam     DT     Input data type
/// @tparam     RT     Output data type
///
/// @return     Truncated value
///
template <typename DT, typename RT>
inline DT truncate(const DT& value, const RT* range = RANGE_8) {
    DT new_value = value;
    if (value < range[0]) {
        new_value = range[0];
    }
    if (value > range[1]) {
        new_value = range[1];
    }
    return new_value;
}

//------------------------------------------------------------------------------
// convert unary intensity value to multi-channel color descriptor
//
// @param[in]  value    The value
// @param[in]  channel  The channel
// @param[in]  range    The range
//
// @return     { description_of_the_return_value }
//
inline Scalar toColor(float value, int channel, const float* range = RANGE_8) {
    // truncate value to match range
    value = truncate(value, range);
    if (channel == 0) {
        return Scalar(value, 0, 0, 0);
    }
    if (channel == 1) {
        return Scalar(0, value, 0, 0);
    }
    if (channel == 2) {
        return Scalar(0, 0, value, 0);
    }
    if (channel == 3) {
        return Scalar(0, 0, 0, value);
    }
    throw std::invalid_argument("intensityToColor(): invalid channel");
}
//////////////////////////////////////////////////////////////////////////////////////////////
// check if val fits input range
template < typename T >
inline bool inRange(const T& val, const ColorRange& range) {
    if (val > range[0] && val < range[1]) {
        return true;
    }
    return false;
}
//////////////////////////////////////////////////////////////////////////////////////////
// circular point vector
// requires center point and radius, number of output points can also be adapted, defaults to 20
inline std::vector< Point > circumference(const Point& center, float radius, size_t n_points = 20) {
    // arc angle between points
    float angle_step = 2 * M_PI / n_points;
    std::vector< Point > circle_points;
    // for all other points, compute arc length
    float angle = 0.0;
    for (int i = 0; i < n_points - 1; ++i) {
        angle += angle_step;
        circle_points.emplace_back(center.x + radius * cos(angle), center.y + radius * sin(angle));
    }
    return circle_points;
}
//////////////////////////////////////////////////////////////////////////////////////////
// constructs a [square] Rect2d object  from from center point and side lenght
// alternative to default Rect constructor, for conveninc
inline Rect2d square(const Point& center, float side_lenght) {
    return cv::Rect2d(center.x - 0.5 * side_lenght, center.y - 0.5 * side_lenght, side_lenght, side_lenght);
}
//////////////////////////////////////////////////////////////////////////////////////////
// constructs a Point vector from existing Rect object
// NOTE: template needs to be explicitly specialized when calling the function, return types aren't deduced implicitly
template < typename T >
inline std::vector< Point_<T> > toPoints(const Rect& rectangle, const Point_<T>& offset = Point()) {
    std::vector< Point_<T> > rectangle_points(4);
    rectangle_points[0] = offset + Point_<T>(rectangle.x, rectangle.y);                                       // top left corner
    rectangle_points[1] = offset + Point_<T>(rectangle.x + rectangle.width, rectangle.y);                     // top right corner
    rectangle_points[2] = offset + Point_<T>(rectangle.x + rectangle.width, rectangle.y + rectangle.height);  // bottom right corner
    rectangle_points[3] = offset + Point_<T>(rectangle.x, rectangle.y + rectangle.height);                    // bottom left corner
    return rectangle_points;
}
//////////////////////////////////////////////////////////////////////////////////////////
// analogous version for RotatedRect objects (wraps around .points() member)
template < typename T >
inline std::vector< Point_<T> > toPoints(const RotatedRect& rectangle, const Point_<T>& offset = Point()) {
    std::vector< Point_<T> > rectangle_points(4);
    rectangle.points(rectangle_points.data());
    for (Point_<T>& point : rectangle_points) {
        point += offset;
    }
    return rectangle_points;
}
//////////////////////////////////////////////////////////////////////////////////////////
// IMAGE TRANSFORMATION
//////////////////////////////////////////////////////////////////////////////////////////
// stitch two frames together, horizontally or vertically
inline void stitch(Mat* target_ptr, const Mat& frame, const Mat& other_frame, bool horizontal = true, size_t padding = 0) {
    // check input frame types
    if (frame.type() != other_frame.type()) {
        throw std::invalid_argument("stitch(): input images of different type;");
    }
    // calc new frame dimensions
    int cols = (horizontal ? (frame.cols + padding + other_frame.cols) : max(frame.cols, other_frame.cols));
    int rows = (horizontal ? max(frame.rows, other_frame.rows) : (frame.rows + padding + other_frame.rows));
    *target_ptr = Mat::zeros(rows, cols, frame.type());
    // root point for first input frame
    Point idx(0,0);
    // root point for second frame (depends on horizontal/vertical stitching)
    Point other_idx(0,0);
    other_idx.x = horizontal ? (frame.cols + padding) : 0;
    other_idx.y = horizontal ? 0 : (frame.rows + padding);
    // copy first frame
    // loop over rows individually)
    for (int row = 0; row < frame.rows; row++) {
        // NOTE: fastest way to loop over a Mat is to use row pointers
        // NOTE: may need to change acess template from Vec3b to uchar
        // if (frame.channels() == 1) {
        //      // ...
        // }
        auto row_ptr        = frame.ptr< Vec3b >(row);
        auto target_row_ptr = target_ptr->ptr< Vec3b >(row + idx.y);
        // loop over cols
        for (int col = 0; col < frame.cols; col++) {
            target_row_ptr[col+ idx.x] = row_ptr[col];
        }
    }
    for (int row = 0; row < other_frame.rows; row++) {
        auto row_ptr        = other_frame.ptr< Vec3b >(row);
        auto target_row_ptr = target_ptr->ptr< Vec3b >(other_idx.y + row);
        // loop over cols
        for (int col = 0; col < other_frame.cols; col++) {
            target_row_ptr[other_idx.x + col] = row_ptr[col];
        }
    }
}
// allocator overload (returns new Mat)
inline Mat stitch(const Mat& frame, const Mat& other_frame, bool horizontal = true, size_t padding = 0) {
    Mat stitched;
    stitch(&stitched, frame, other_frame, horizontal, padding);
    return stitched;
}
//////////////////////////////////////////////////////////////////////////////////////////
// plot line on image
// draws in place in order to overlay multiple lines on the same without passing Mat copies around
// useful to plot multiple unidimensional (single channel) histograms on the same image directly
inline void plotLine(Mat* target_ptr, const Mat& data, const Scalar& color = Scalar(255, 0, 0), int thickness = 2, bool norm = true) {
    // normalize input data in order to ensure it fits image dimensions
    // Mat normalized_data;
    if (norm == true) {
        // normalize(data, normalized_data, 0, image->rows, NORM_MINMAX, -1, cv::Mat());
        normalize(data, data, 0, target_ptr->rows, NORM_MINMAX, -1, cv::Mat());
    }
    // calc bin  width
    int bin_w = cvRound((double) target_ptr->cols / data.rows);
    // int bin_w = static_cast<int>((double) image->cols/normalized_data.rows);
    // draw lines between each input data value
    for( int i = 1; i < data.rows; i++) {
        line(*target_ptr,
             Point(bin_w * (i-1), target_ptr->rows - cvRound(data.at<float>(i-1))), /* origin */
             Point(bin_w * (i),   target_ptr->rows - cvRound(data.at<float>(i)) ),  /* end*/
             color, thickness, 8, 0);
    }
}
// allocator overload (returns new Mat w/ line plotted)
inline Mat plotLine(const Mat& data, const Scalar& color = Scalar(255, 0, 0), int thickness = 2, bool norm = true, Size size = { 500, 500 }) {
    Mat plotted(size, CV_8UC1, Scalar::all(0));
    plotLine(&plotted, data, color, thickness, norm);
    return plotted;
}
//////////////////////////////////////////////////////////////////////////////////////////
// plot 2D data onto existing image
// draws in place in order to overlay multiple lines on the same without passing Mat copies around
// useful to plot multiple 2D (double channel) histograms on the same image directly
inline void plotMat(Mat* target_ptr, const Mat& data, size_t channel, float scale = 1.0, const float* range = RANGE_8) {
    if (channel >= target_ptr->channels()) {
        throw std::runtime_error("plotMat(): invalid channel index");
    }
    // scales for row and col index
    float row_scale = (target_ptr->rows + 0.0) / data.rows;
    float col_scale = (target_ptr->cols + 0.0) / data.cols;
    // find max and min value
    double min, max;
    Point min_loc, max_loc;
    minMaxLoc(data, &min, &max, &min_loc, &max_loc);
    // draw values as 'squares'/pixels scaled to iamge size
    float val;
    float intensity;
    for (int row = 0; row < data.rows; row++) {
        for (int col = 0; col < data.cols; col++) {
            // get value
            val = data.at< float >(row, col);
            // normalize value (scale color to 8bit)
            intensity = truncate(cvRound(scale * (val / max) * range[1]), range);
            // draw pseudo-pixel as a rectangle to scaled positions
            rectangle(*target_ptr,
                      Point(cvRound(col * col_scale), cvRound(row * row_scale)),                      /* left upper corner   */
                      Point(cvRound((col + 1) * col_scale - 1), cvRound((row + 1) * row_scale - 1)),  /* right bottom corner */
                      toColor(intensity, channel, range), -1);
        }
    }
}
// allocator overload (returns new Mat w/ Mat plotted)
inline Mat plotMat(const cv::Mat& data, size_t channel, float scale = 1.0, const float* range = RANGE_8, Size size = { 500, 500 }) {
    Mat plotted(size, CV_8UC1, Scalar::all(0));
    plotMat(&plotted, data, channel, scale, range);
    return plotted;
}
//////////////////////////////////////////////////////////////////////////////////////////
// overlay/draw point vector onto target image
// template allows using different types of point cooridates (e.g. Point2d, Point2i, etc)
template < typename T >
inline void drawShape(Mat* target_ptr, const std::vector< Point_<T> >& points, const Scalar& color, int thickness = 2) {
    // parse arguments
    if (points.size() <= 2) {
        throw std::invalid_argument("");
    }
    // draw line between last and first points
    line(*target_ptr, points.back(), points.front(), color, thickness, 8, 0);
    // loop over remaining points, draw lines between them
    for (int point_idx = 0; point_idx < points.size() - 1; point_idx++) {
        line(*target_ptr,
             points[point_idx],
             points[point_idx + 1],
             color, thickness, 8, 0);
    }
    // what happens if point falls outside image boundaries?
    // what happens if color does not match channels?
}
// allocator overload (returns new Mat)
template < typename T >
inline Mat drawShape(const std::vector< Point_<T> >& points, const Scalar& color, int thickness = 2, Size size = { 500, 500 }) {
    Mat drawn(size, CV_8UC1, Scalar::all(0));
    drawShape(&drawn, points, color, thickness);
    return drawn;
}
//////////////////////////////////////////////////////////////////////////////////////////////
// HISTOGRAM COMPUTATION AND MANAGEMENT
//////////////////////////////////////////////////////////////////////////////////////////////
// DEPRECATED, USE histogramND(...) INSTEAD
// TODO(joao): add allocator overload
// 1D histogram computation
// defaults to first channel, 8bit range (0-255)
// assumes 1 bin per integer value, acording to input range e.g. 255 bins if range_X == {0, 255}
// no check is done if values match the input range, it's up to the caller to ensure that
inline Mat histogram1D(const Mat&   frame,
                       const Mat&   mask,
                       int          channel = 0,
                       const float* range   = RANGE_8) {
    // make sure channels have proper values
    if (channel < 0 && channel >= frame.channels() ) {
        throw std::invalid_argument("histogram1D(): invalid channel index");
    }
    // parse input argumnts
    int channels[]        = { channel };
    int n_bins[]          = { static_cast<int>(range[1] - range[0]) };
    const float* ranges[] = { range };
    // compute histogram (wrap around cv::calcHist)
    Mat histogram;
    try {
        calcHist(&frame, 1, channels, mask, histogram, 1, n_bins, ranges, true, false);
    } catch (Exception& error) {
        throw std::runtime_error("histogram1D(): error computing histogram, check argument values");
    }
    return histogram;
}
//////////////////////////////////////////////////////////////////////////////////////////////
// DEPRECATED, USE histogramND(...) INSTEAD
// TODO(joao): add allocator overload
// 2D histogram calculation
// wraps around cv::calcHist to provide a simpler interface
// assumes 1 bin per integer value, acording to input range e.g. 255 bins if range_X == {0, 255}
// no check is done if values match the input range, it's up to the caller to ensure that
// TODO(joao): change argument types to std::vector<>
inline Mat histogram2D(const Mat& frame,
                       const Mat& mask,
                       int        channel_0,
                       int        channel_1,
                       ColorRange range_0 = RANGE_8,
                       ColorRange range_1 = RANGE_8) {
    // make sure channels have proper values
    if (channel_0 < 0 || channel_0 >= frame.channels() || channel_1 < 0 || channel_1 >= frame.channels()) {
        throw std::invalid_argument("histogram2D(): invalid channel index");
    }
    // parse input argumnts
    int channels[]        = { channel_0, channel_1 };
    int n_bins[]          = { static_cast<int>(range_0[1] - range_0[0]), static_cast<int>(range_1[1] - range_1[0]) };
    const float* ranges[] = { range_0, range_1 };
    // compute histogram (wrap around cv::calcHist)
    Mat histogram;
    try {
        calcHist(&frame, 1, channels, mask, histogram, 2, n_bins, ranges, true, false);
    } catch (Exception& error) {
        throw std::runtime_error("histogram2D(): error computing histogram, check argument values");
    }
    return histogram;
}
//////////////////////////////////////////////////////////////////////////////////////////////
// multidimensional histogram computation
// accepts vector of channel indexes and ranges (dimensions should match)
// assumes 1 bin per integer level (ranges == bins)
inline void histogramND(Mat*       target_ptr,
                        const Mat& source,
                        const Mat& mask,
                        std::vector< int >        channels,
                        std::vector< ColorRange > ranges) {
    // parse input arguments
    if (channels.size() > source.channels()) {
        throw std::invalid_argument("histogramND(): invalid channel vector (> frame.channels())");
    }
    for (auto& channel : channels) {
        if (channel >= source.channels()) {
            throw std::invalid_argument("histogramND(): invalid channel value (> frame.channels())");
        }
    }
    if (ranges.size() != channels.size()) {
        throw std::invalid_argument("histogramND(): invalid range vector (!= channels.size())");
    }
    // populate bins vector -> 1 bin per integer level
    // NOTE: only tested with CV_8U Mat types
    std::vector< int > n_bins;
    for (auto& range : ranges) {
        n_bins.emplace_back(static_cast<int>(range[1] - range[0] + 1));
    }
    // compute histogram (wrap around cv::calcHist)
    try {
        calcHist(&source, 1, channels.data(), mask, *target_ptr, ranges.size(), n_bins.data(), ranges.data(), true, false);
    } catch (cv::Exception& error) {
        throw std::runtime_error("histogramND(): error computing histogram, check argument values");
    }
}
// allocator overload (returns new Mat)
inline Mat histogramND(const Mat& source, const Mat& mask, std::vector< int > channels, std::vector< ColorRange > ranges) {
    Mat histogram;
    histogramND(&histogram, source, mask, channels, ranges);
    return histogram;
}
//////////////////////////////////////////////////////////////////////////////////////////
// histogram's cumulative distribution function from pre-computed histogram
// only works with 1D histograms, will throw exception otherwise
// output is a 32-bit one-dimensional array with (normalized) pixel count of each intensity level
// range is the same as input histogram array, so it works with 8-bit
// NOTE: does not work inplace i.e. target_ptr must not point to source histogram argument
// TODO(joao): swap loop direction (-> backwards), be able to modify in place!
inline void histogramCDF(Mat* target_ptr, const Mat& histogram, bool norm = false) {
    // check if one dimensional histogram
    if (histogram.cols > 1) {
        throw std::invalid_argument("histogramCDF(): invalid input histogram (cols > 1)");
    }
    // initialize output array with same size, 0 values
    // NOTE: previous content of target_ptr image is discarded
    *target_ptr = Mat(histogram.size(), CV_32FC1, cv::Scalar(0, 0, 0, 0));
    for (int r = 0; r < histogram.rows; r++) {
        for (int p = 0; p <= r; p++) {
            target_ptr->at< float >(r) += histogram.at< float >(p);
        }
    }
    // normalization step to 0-1 range
    // NOTE: depth of returning Mat becomes CV_32F
    // NOTE 2: no need to normalize in bulk, may b faster to just divide by total pixel account on first loop
    if (norm) {
        normalize(*target_ptr, *target_ptr, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    }
}
// allocator overload (returns new Mat)
inline Mat histogramCDF(const Mat& histogram, bool norm = false) {
    Mat histogram_cdf;
    histogramCDF(&histogram_cdf, histogram, norm);
    return histogram_cdf;
}
//////////////////////////////////////////////////////////////////////////////////////////
// direct CDF calculation from input image
// wraps around histogramND(...) and histogramCDF(...) for convenience,
inline void cumulativeHistogramFromSource(Mat* target_ptr, const Mat& source, const Mat& mask, std::vector< int > channels, std::vector< ColorRange > ranges, bool norm = false) {
    Mat histogram;
    histogramND(&histogram, source, mask, channels, ranges);
    // histogramCDF(histogram, histogram, norm); // DOES NOT WORK
    histogramCDF(target_ptr, histogram, norm);
}
// allocator overload, wraps around histogramCDF and histogramND
inline Mat cumulativeHistogramFromSource(const Mat& source, const Mat& mask, std::vector< int > channels, std::vector< ColorRange > ranges, bool norm = false) {
    // alternatively, cumulativeHistogramFromSource(...) could be called
    return histogramCDF(histogramND(source, mask, channels, ranges), norm);
}
//////////////////////////////////////////////////////////////////////////////////////////
// interpolate index between values
// returns estimated index (float, continuous) of input value
// input matrix should be one dimensional (.cols() == 1), of type <T> and sorted low->high(
// TODO(joao) -> do I need to template this?
template < typename T>
inline void estimateIndex(float* target_ptr, const Mat& source, T val) {
    // parse input source array -> should be one dimensional and sorted ascendingly
    if (source.cols > 1) {
        throw std::invalid_argument("estimateIndex(): invalid dimensions on source array.");
    }
    if (source.at< T >(0) > source.at< T >(1)) {
        // may be needed to check this while looping over input data, first two values can  be th only ones to be sorted
        throw std::invalid_argument("estimateIndex(): source array is not sorted.");
    }
    int idx = -1;
    int lower_idx = 0;
    int upper_idx = 0;
    // find fist source value greater than input val
    while (source.at< T >(idx + 1) < val) { // && idx < source.rows) {
        idx++;
        // if end of array is reached
        // NOTE: inside loop vs on stop condition to avoid acessing source.at< T >(source.rows)
        if (idx > (source.rows -1)) {
           break;
        }
    }
    // assign lower and upper index bounds
    if (idx == -1) {
        // first value is already greater than input, first two values used
        lower_idx = 0;
        upper_idx = 1;
    } else if (idx == source.rows) {
        // last value still smaller, last two values used
        lower_idx = source.rows - 2;
        upper_idx = source.rows - 1;
    } else {
        lower_idx = idx;
        upper_idx = idx + 1;
    }
    // compute slope between lower and upper boundary -> derivative for value interpolation
    float f = (source.at< T >(upper_idx) - source.at< T >(lower_idx)); //  / (upper_idx - lower_idx);
    if (f == 0.0) {
        *target_ptr = 0.0;
    }
    *target_ptr = lower_idx + ((val - source.at< T >(lower_idx)) / f);
}
// allocator overload
template < typename T>
inline float estimateIndex(const Mat& source, T val) {
    float index;
    estimateIndex(&index, source, val);
    return index;
}
//////////////////////////////////////////////////////////////////////////////////////////
// computes intensity level matchings between two different histograms
// takes histograms as arguments, makes no assumption on the image type or range
// returns 8bit array look-up table (cv::LUT only works with 8bit input arrays)
// both input histograms should match range i.e. same dimensions
// TODO(joao): extend applicability to 16-bit/32-bit return types?
// TODO(joao): pass the histogram CDF dirctly as arguments?
// TODO(joao): do inputs really need to be of same size?
// TODO(joao): use same return type as input histograms, no ned to use an argument for that (should only allow 8bit anyway)
inline void histogramMatchLUT(Mat* target_ptr, const Mat& histogram, const Mat& other_histogram, int return_type = CV_8UC1) {
    // parse arguments
    if (histogram.rows != other_histogram.rows) {
        throw std::invalid_argument("histogramMatch(): input arrays do not match.");
    }
    // compute cumulative histograms (normalize == true)
    Mat cdf       = histogramCDF(histogram, true);
    Mat other_cdf = histogramCDF(other_histogram, true);
    // initialize target image (same size )
    *target_ptr = Mat(histogram.size(), return_type);
    // for each bin on first image, find matching bin on other image (round estimated index to integer)
    for (int bin_idx = 0; bin_idx < cdf.rows; bin_idx++) {
        // assign value as the correspondent new value on the original histogram
        // gets index value of cdf value, which translates to matching intensity level
        int other_bin = cvRound(estimateIndex(other_cdf, cdf.at< float >(bin_idx)));
        // float idx = estimateIndex(other_cdf, cdf.at< float >(bin_idx));
        // int low = (int)(idx);
        // int high = std::ceil(idx);
        // printf("debug4 %d // %f \n", other_bin, idx);
        // std::cout << cdf.at< float >(bin_idx) << " [@ " << bin_idx << "] matches " << other_cdf.at< float >(other_bin) << " [@ " << other_bin << "], between " << other_cdf.at< float >(low) << " [@ " << low << "] and " << other_cdf.at< float >(high) << " [@ " << high << "] -> [@ " << idx << "]" << std::endl;
        target_ptr->at< uint8_t >(bin_idx) = other_bin;
        // std::cout << cdf.at< float >(bin_idx) << " (" << bin_idx << ") -> " << other_cdf.at< float >(other_bin) << " (" << other_bin << ")" << std::endl;
        // Mat debug2 = debug;
        // line(debug2,
             // Point(bin_idx * scale, (256 - cdf.at< float >(bin_idx) * 256) * scale), /* origin */
             // Point(other_bin * scale,( 256 - other_cdf.at < float >(other_bin) * 256) * scale),  /* end*/
             // Scalar(0, 0, 255), 2, 8, 0);
        // printf("done\n");
        // imshow("debug", debug2);
        // waitKey(0);
    }
}
// overload returning a new copy
inline Mat histogramMatchLUT(const Mat& histogram, const Mat& other_histogram, int return_type = CV_8UC1) {
    Mat lut;
    histogramMatchLUT(&lut, histogram, other_histogram, return_type);
    return lut;
}
//////////////////////////////////////////////////////////////////////////////////////////
// apply single_channel histogram matching to image directly
// wraps around histogramND(...), histogramCDF(...) and histogramMatchLUT(...)
inline void matchChannelHistogram(Mat*       target_ptr,
                                  const Mat& source,
                                  const Mat& reference_histogram,
                                  const Mat& mask,
                                  int        channel,
                                  ColorRange range) {
    // NOTE: reference_histogram should have the same
    if (reference_histogram.cols > 1) {
        throw std::invalid_argument("matchChannelHistogram(): Invalid reference histogram ");
    }
    // 1. compute histogram and cumulative (all equivalent methods)
    // since only 1D histogram, simpler to use histogram1D
    // Mat source_cdf = histogramCDF(histogramND(source, mask, { channel }, { range }), true);
    // Mat source_cdf = cumulativeHistogramFromSource(source, mask, { channel }, { range }, true);
    Mat source_cdf = histogramCDF(histogram1D(source, mask, channel, range), true);
    // 2. get histogram match
    Mat matching_lut = histogramMatchLUT(source_cdf, reference_histogram);
    // apply LUT to source image
    // NOTE: cv::LUT only allows 8bit images and 256 unidimenstional lookup tables
    LUT(source, matching_lut, *target_ptr);
}
// allocator overload returning a new Mat
inline Mat matchChannelHistogram(const Mat& source,
                                 const Mat& reference_histogram,
                                 const Mat& mask,
                                 int        channel,
                                 ColorRange range) {
    Mat matched;
    matchChannelHistogram(&matched, source, reference_histogram, mask, channel, range);
    return matched;
}
//////////////////////////////////////////////////////////////////////////////////////////////
// returns cropped image, according to the specified ROI vertices
// TODO(joao): add invert flag for concave cropping (pixels outside ROI)
// NOTE: IT DOES NOT WORK IN-PLACE!





//------------------------------------------------------------------------------
/// @brief      Thresholds (& crops) input image to ROI
///
/// @param      target_ptr  Target image
/// @param[in]  frame       Input image
/// @param[in]  points      ROI corners
/// @param[in]  crop        Crop flag.
///
/// @tparam     PT          ROI point data type
/// @tparam     T           Output image data type
///
template < typename PT, typename T >
inline void toROI(Mat* target_ptr, const Mat& frame, const std::vector< Point_< T > >& points, bool crop = false) {
    // parse point vector argument
    if (points.size() < 3) {
        throw std::invalid_argument("toROI(): Invalid ROI point vector (size != 4)");
    }
    // get min and max row and col
    // required to find size of cropped image
    int min_row = frame.rows;
    int min_col = frame.cols;
    int max_row = 0;
    int max_col = 0;
    if (crop == true) {
        for (auto& point : points) {
            // if (point.x >= frame.cols || point.y >= frame.rows) {
            //     throw std::invalid_argument("cropToROI(): Invalid ROI vertex point (outside image boundary)");
            // }
            if (point.x < min_col) {
                min_col = point.x;
            }
            if (point.y < min_row) {
                min_row = point.y;
            }
            if (point.x > max_col) {
                max_col = point.x;
            }
            if (point.y > max_row) {
                max_row = point.y;
            }
        }
    } else {
        min_row = 0;
        min_col = 0;
        max_row = frame.rows;
        max_col = frame.cols;
    }
    // note: if crop == false, then height and width default to input frame dimensions
    int height = abs(max_row - min_row);
    int width  = abs(max_col - min_col);
    // instantiate new image
    if (target_ptr != &frame) {
        *target_ptr = Mat(height, width, frame.type(), Scalar::all(0));
    }
    // loop over input frame pixels
    for (size_t row_idx = 0; row_idx < frame.rows; row_idx++ ) {
        for (size_t col_idx = 0; col_idx < frame.cols; col_idx++ ) {
            Point2i pixel(col_idx, row_idx);

            if (pixel.x >= frame.cols || pixel.y >= frame.rows) {
                continue;
            }
            // check if inside ROI
            if (pointPolygonTest(points, pixel, false /* binary classification, no need to measure distance */) >= 0) {
                // check if point exists in original image
                // required in cases where ROI points are outside dimensions
                // alternatively, a check could be performed when parsing input, would avoid an additional check for very pixel
                // (note: difference should not be noticeable and it adds versatility to the function)
                if (target_ptr != &frame) {
                    target_ptr->at< PT >(row_idx - min_row, col_idx - min_col) = frame.at< PT >(row_idx, col_idx);
                }
                // // copy pixel value to new image
                // if (frame.channels() > 1) {
                //     // multi channel image (e.g. BGR, HSV, etc)
                //     target_ptr->at< cv::Vec3b >(row_idx - min_row, col_idx - min_col) = frame.at< cv::Vec3b >(row_idx, col_idx);
                // } else {
                //     // single channel images (e.g. grayscale, etc)
                //     target_ptr->at< uint8_t >(row_idx - min_row, col_idx - min_col) = frame.at< uint8_t >(row_idx, col_idx);
                // }
            } else if (target_ptr == &frame) {
                target_ptr->at< PT >(row_idx - min_row, col_idx - min_col) = static_cast< PT >(0);
            }
        }
    }
}
// allocator overload
// NOTE: template allows any type of Point vector
template < typename PT, typename T >
inline Mat toROI(const Mat& frame, const std::vector< Point_< T > >& points, bool crop = false) {
    Mat roi;
    toROI< PT >(&roi, frame, points, crop);
    return roi;
}
// allocator overload
// NOTE: template allows any type of Point vector
template < typename PT, typename T >
inline void toROI(Mat* frame, const std::vector< Point_< T > >& points, bool crop = false) {
    toROI< PT >(frame, *frame, points, crop);
}
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
// sets target channel to input Mat
inline void setChannel(Mat* target_ptr, size_t channel_idx, const Mat& channel_data) {
    if (target_ptr->channels() < channel_idx) {
        throw std::invalid_argument("setChannel(): invalid channel index;");
    }
    if (channel_data.channels() > 1 || channel_data.size() != target_ptr->size()) {
        throw std::invalid_argument("setChannel(): invalid input data, it should match target size;");
    }
    // method 1 -> loop over all pixels
    // cv::Vec3b value;
    // for (int r = 0; r < target->rows; r++) {
    //     for (int c = 0; c < target->cols; c++) {
    //         value = target->at<cv::Vec3b>(r,c);
    //         value[channel_idx] = channel_data.at<uint8_t>(r, c);
    //     }
    // }
    // method 2 -> bulk assignment -> is it more efficient?
    std::vector< Mat > target_channels;
    split(*target_ptr, target_channels);
    target_channels[channel_idx] = channel_data;
    try {
        merge(target_channels, *target_ptr);
    } catch (Exception& error) {
        throw std::invalid_argument("setChannel(): error @ cv::merge -> are inputs of the same type?");
    }
}
// overload for scalar assignment (8bit only -> add overloads or template if other types are required)
// note: a more efficient way would be to loop over all pixels and set channel to the value
inline void setChannel(Mat* target_ptr, size_t channel_idx, const uchar& value) {
    setChannel(target_ptr, channel_idx,  cv::Mat(target_ptr->size(), CV_8UC1, cv::Scalar(value)));
}
//
inline Mat setChannel(size_t channel_idx, const Mat& channel_data) {
    Mat out;
    setChannel(&out, channel_idx, channel_data);
    return out;
}
//
inline Mat setChannel(size_t channel_idx, const uchar& value) {
    Mat out;
    setChannel(&out, channel_idx, value);
    return out;
}
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace cv
//////////////////////////////////////////////////////////////////////////////////////////////
#endif  // INCLUDE_CVUTILS_HPP
