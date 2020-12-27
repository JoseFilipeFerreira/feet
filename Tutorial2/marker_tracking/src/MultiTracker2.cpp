#include <string>
#include <vector>
#include "CVUtils.hpp"
#include "MultiTracker2.hpp"
#include "MouseCoordinateSaver.hpp"
//////////////////////////////////////////////////////////////////////////////////////////
namespace cv {
//////////////////////////////////////////////////////////////////////////////////////////
// empty/default constructor from number of marker
// marker shapes/bounding boxes initialized as empty vector
MultiTracker2::MultiTracker2(size_t n_markers, const std::string& algorithm) :
    _algorithm(algorithm),
    _marker_boxes(n_markers), 
    _tracker_ptrs(n_markers) { 
        // can't use std::vector single value constructor overload as it creates copies of the smart pointer, for the same underlying object
        // for (int idx = 0; idx < n_markers; ++idx) {
        //     _tracker_ptrs[idx] = MultiTracker2::createTracker(algorithm);
        // }
}
// constructor overload for existing marker boxes
// initializes trackers
MultiTracker2::MultiTracker2(const std::vector< MarkerBox >& marker_boxes, const std::string& algorithm) :
    _marker_boxes(marker_boxes),
    _tracker_ptrs(marker_boxes.size()) {
        // can't use std::vector single value constructor overload as it creates copies of the smart pointer, for the same underlying object
        for (int idx = 0; idx < marker_boxes.size(); ++idx) {
            _tracker_ptrs[idx] = MultiTracker2::createTracker(algorithm);
        }
        // NOTE: trackers still need to be initialized with init()
}
size_t MultiTracker2::nTrackers() const {
    return _tracker_ptrs.size();
}
bool MultiTracker2::hasData() const {
    return (_marker_boxes.size() > 0);
}
const std::vector< MultiTracker2::TrackerPtr >& MultiTracker2::trackers() const {
    return _tracker_ptrs;
}
const std::vector< MultiTracker2::MarkerBox >& MultiTracker2::markers() const {
    return _marker_boxes;
}
void MultiTracker2::init(const Mat& frame, size_t bounding_box_size, int marker_idx) {
    MouseCoordinateSaver marker_saver(frame); // , _tracker_ptrs.size());
    if (marker_idx < 0) {
        // prompt user to select marker
        marker_saver.get("Identify markers", nTrackers());
        // instantiate and init trackers
        for (int idx = 0; idx < nTrackers(); idx++) {
            _marker_boxes[idx] = cv::square(marker_saver.coordinates()[idx], bounding_box_size);

            _tracker_ptrs[idx] = MultiTracker2::createTracker(_algorithm);
            _tracker_ptrs[idx]->init(frame, _marker_boxes[idx]);
        }
    } else if (marker_idx < nTrackers()){
        // prompt user to select marker
        marker_saver.get("Identify markers", 1);
        // construct bounding box from saved coordinates
        _marker_boxes[marker_idx] = cv::square(marker_saver.coordinates()[0], bounding_box_size);
        // init tracker
        _tracker_ptrs[marker_idx] = MultiTracker2::createTracker(_algorithm);
        _tracker_ptrs[marker_idx]->init(frame, _marker_boxes[marker_idx]);
    }
    // if (1) {
    // // if (!hasData()) {
    //     // prompt user selection of marker coordinates
    //     MouseCoordinateSaver marker_saver(frame); // , _tracker_ptrs.size());
    //     marker_saver.get("Identify markers", nTrackers());
    //     // construct bounding boxes
    //     for (auto& center : marker_saver.coordinates()) {
    //         _marker_boxes.emplace_back(cv::square(center, bounding_box_size));
    //         // printf("%d\n", bounding_box_size);
    //     }
    // }
    // init trackers
    // for (int idx = 0; idx < nTrackers(); idx++) {
    //     _tracker_ptrs[idx]->init(frame, _marker_boxes[idx]);
    // }
}
// NOTE: does instantiating and returning a std::vector< bool > has performance impact?
std::vector< bool > MultiTracker2::update(const Mat& frame) {
    std::vector< bool > sucess(nTrackers());
    for (int idx = 0; idx < nTrackers(); idx++) {
        // printf("Using tracker @%p to track box at %f, %f\n", _tracker_ptrs[idx].get(), _marker_boxes[idx].x, _marker_boxes[idx].y);
        sucess[idx] = _tracker_ptrs[idx]->update(frame, _marker_boxes[idx]);
    }
    return sucess;
}
//////////////////////////////////////////////////////////////////////////////////////////
// allocate a cv::Tracker object on the heap, according to input algorithm name
cv::Ptr< cv::Tracker > MultiTracker2::createTracker(const std::string& algorithm) {
    if (algorithm == "BOOSTING") {
        return cv::TrackerBoosting::create();
    } else if (algorithm == "MIL") {
        return cv::TrackerMIL::create();
    } else if (algorithm == "KCF") {
        return cv::TrackerKCF::create();
    } else if (algorithm == "TLD") {
        return cv::TrackerTLD::create();
    } else if (algorithm == "MEDIANFLOW") {
        return cv::TrackerMedianFlow::create();
    } else if (algorithm == "GOTURN") {
        return cv::TrackerGOTURN::create();
    } else if (algorithm == "MOSSE") {
        return cv::TrackerMOSSE::create();
    } else if (algorithm == "CSRT") {
        return cv::TrackerCSRT::create();
    }
    return cv::Ptr< cv::Tracker>();
}
//////////////////////////////////////////////////////////////////////////////////////////
// overlay marker boxes on a target image, according to success vector
// @todo    use color vector instead of success
void MultiTracker2::drawMarkers(cv::Mat* target, const std::vector< MarkerBox >& marker_boxes, const std::vector< bool > success) {
    if (marker_boxes.size() != success.size()) {
        throw std::invalid_argument("MultiTracker2::drawMarkers(): invalid sucess vector (size != marker_boxes)");
    }
    for (uint idx = 0; idx < marker_boxes.size(); ++idx) {
        if (success[idx] == true) {
            cv::rectangle(*target, marker_boxes[idx], Scalar(0, 255, 0) /* green */, 2, 1);  
        } else {
            cv::rectangle(*target, marker_boxes[idx], Scalar(0, 0, 255) /* red */, 2, 1);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
}  // namespace cv
//////////////////////////////////////////////////////////////////////////////////////////