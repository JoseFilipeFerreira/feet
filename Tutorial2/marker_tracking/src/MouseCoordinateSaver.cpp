#include <vector>
#include <exception>
#include <opencv2/highgui.hpp>    // cv::setMouseCallback
#include "MouseCoordinateSaver.hpp"
//////////////////////////////////////////////////////////////////////////////////////////////
#define KEY_PRESS_TIMEOUT_MS 1   // timeout for key check, in ms // added to variable delay inferred from frame_rate parameter
#define CLOSE_KEY            27  // 'ESC'
//////////////////////////////////////////////////////////////////////////////////////////////
namespace cv {
//////////////////////////////////////////////////////////////////////////////////////////////
MouseCoordinateSaver::MouseCoordinateSaver(const Mat& frame) :
    _frame_ptr(&frame) {
        /* ... */
}
MouseCoordinateSaver::~MouseCoordinateSaver() {
    /* ... */
}
const std::vector< Point>& MouseCoordinateSaver::coordinates() const {
    return _coordinates;
}   
const Point& MouseCoordinateSaver::coordinates(size_t index) const {
    if (index >= _coordinates.size()) {
        throw std::invalid_argument("MouseCoordinateSaver::coordinates(): invalid point index");
    }
    return _coordinates[index];
}
// TODO(joao) : change return type to allow single channel and multi channel frames
// maybe Vec_< frame.channels >???
template < typename T >
T MouseCoordinateSaver::value(size_t index) const {
    if (index >= _coordinates.size()) {
        throw std::invalid_argument("MouseCoordinateSaver::value(): invalid point index");
    }
    return _frame_ptr->at< Vec3b >(_coordinates[index].y, _coordinates[index].x); // opencv is row-major ! 
}

// explicit specialization of templated function
// template < > float    MouseCoordinateSaver::value< float >(size_t);     // depth/numeric matrices
// template < > Vec3b    MouseCoordinateSaver::value< Vec3b >(size_t);     // BGR images
// template < > uint16_t MouseCoordinateSaver::value< uint16_t >(size_t);  // depth images

// NOTE: coordinate vector is not cleared to allow separate coordinate saving to same list 
void MouseCoordinateSaver::get(const std::string& window_name, uint n_coords) {
    // open window
    imshow(window_name, *_frame_ptr);
    // set callback function (static member)
    setMouseCallback(window_name, &MouseCoordinateSaver::addCoordinates, this);
    // wait for target positions to be reached
    // checks every 1 ms if abort key has been pressed or if number of target coordinates have been saved
    size_t initial_size = _coordinates.size();
    while (_coordinates.size() - initial_size < n_coords) {
        if (waitKey(KEY_PRESS_TIMEOUT_MS) == CLOSE_KEY) {
            // assing empty positions 
            _coordinates = std::vector< Point > (n_coords, Point(0, 0));
            break;
        }
    }
    // close window after selection
    destroyWindow(window_name);
}
void MouseCoordinateSaver::clear() {
    _coordinates.clear();
}
void MouseCoordinateSaver::addCoordinates(int event, int x, int y, int flags, void* saver_ptr) {
    // cast and deref input argument (void* to cv::Mat&)
    MouseCoordinateSaver* coordinate_saver_ptr = (MouseCoordinateSaver*) saver_ptr;
    // get value when left button is down
    if (event == EVENT_LBUTTONDOWN) {  // && coordinate_saver_ptr->_coordinates.size() < coordinate_saver_ptr->_n_targets) {
        // save coordinates
        // NOTE: _coordinates is acessible as a static function is still a class member
        coordinate_saver_ptr->_coordinates.emplace_back(x, y);
        // debug info print
        std::cout << "#" << coordinate_saver_ptr->_coordinates.size() - 1 << ": x = " << x << " y = " << y << std::endl;
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace cv
//////////////////////////////////////////////////////////////////////////////////////////////