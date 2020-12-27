#ifndef _INCLUDE_MULTITRACKER2_HPP_ 
#define _INCLUDE_MULTITRACKER2_HPP_
//////////////////////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>    // cv::Ptr, cv::Rect2d
#include <opencv2/tracking.hpp>  // cv::Tracker
//////////////////////////////////////////////////////////////////////////////////////////////
namespace cv {
// class similar to OpenCV's cv::MultiTracker, but provides more detailed feedback i.e. which 
// trackers are handled individually, while using the same algorithm
// also implemented user-selected 
class MultiTracker2{
 public:
    // public types
    typedef Ptr< Tracker > TrackerPtr;
    typedef Rect2d         MarkerBox;
    // constructor
    MultiTracker2(size_t n_markers, const std::string& algorithm);
    MultiTracker2(const std::vector< MarkerBox >& marker_boxes, const std::string& algorithm);
    // metadata and member object accessors
    size_t  nTrackers() const;
    bool    hasData()   const;
    const std::vector< TrackerPtr >& trackers() const;
    const std::vector< MarkerBox >&  markers()  const;  // provide only centers?
    // tracking management
    void                init(const Mat& frame, size_t bounding_box_size, int marker_idx = -1);
    std::vector< bool > update(const Mat& frame);
    // static factory allocator
    // individually tracker allocation
    static TrackerPtr           createTracker(const std::string& algorithm);
    // static factory allocator
    // templated, overloads each constructor signature
    template< typename... Args > 
    static Ptr< MultiTracker2 > create(Args... args) { return Ptr< MultiTracker2 > (new MultiTracker2(args...)); };
    // static draw
    static void drawMarkers(cv::Mat* target, const std::vector< MarkerBox >& marker_boxes, const std::vector< bool > success);
    // marker box to 3D point converter?
    // static Point3f markerToWorldCoordinates(const MarkerBox& marker_box, const cv::Mat& intrisic_matrix, const cv::Mat& registered_depth_image);

 protected:
    std::string               _algorithm;
    std::vector< TrackerPtr > _tracker_ptrs;
    std::vector< MarkerBox >  _marker_boxes;
};
//////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace cv
//////////////////////////////////////////////////////////////////////////////////////////////
#endif  // _INCLUDE_MULTITRACKER2_HPP_ 
