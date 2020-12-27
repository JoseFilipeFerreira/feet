#ifndef _INCLUDE_MOUSECOORDINATESAVER_HPP_
#define _INCLUDE_MOUSECOORDINATESAVER_HPP_

#include <vector>
#include <opencv2/opencv.hpp>

//------------------------------------------------------------------------------
/// @brief      Generic scope/usage, declared within cv:: namespace
///
namespace cv {

//------------------------------------------------------------------------------
/// @brief      Class that saves pixel coordinates user-set with mouse/trackpad
///
class MouseCoordinateSaver {
 public:
    //--------------------------------------------------------------------------
    /// @brief      Constructs a new instance
    ///
    /// @param[in]  frame  reference to image (cv::Mat) to fetch coordinates from
    ///
    explicit MouseCoordinateSaver(const Mat& frame);

    //--------------------------------------------------------------------------
    /// @brief      Destroys the object
    ///
    virtual ~MouseCoordinateSaver();

    //--------------------------------------------------------------------------
    /// @brief      All saved coordinates
    ///
    /// @return     Read-only reference to vector of coordinates
    ///
    const std::vector< Point2i >& coordinates() const;

    //--------------------------------------------------------------------------
    /// @brief      Coordinates of a specific saved point
    ///
    /// @param[in]  index  Index of the point (as order of saving/mouse click)
    ///
    /// @return     Read-only refence to [index]th point cordinates
    ///
    const Point2i& coordinates(size_t index)   const;

    //--------------------------------------------------------------------------
    /// @brief      Image value at the coordinates of a specific saved point
    ///
    /// @param[in]  index  Index of the point (as order of saving/mouse click)
    ///
    /// @tparam     T      Type of image data
    ///
    /// @return     Instance of T corresponding to image value at [index]th coordinates
    ///
    template < typename T >
    T value(size_t index) const;

    //--------------------------------------------------------------------------
    /// @brief      Prompts user to mouse-select [n_coordinates] points on image
    ///
    /// @param[in]  window_name  Name (text string) to appear on the window title
    /// @param[in]  n_coords     Number of points to save
    ///
    void get(const std::string& window_name, uint n_coords = 1);

    //--------------------------------------------------------------------------
    /// @brief      Clears saved coordinates
    ///
    void clear();

    //--------------------------------------------------------------------------
    // @brief       Event callback, called at every mouse click. Adds coordinates to member vector
    //              Generic function signature to comply with OpenCV's setMouseCallback() function
    //
    // @param[in]  event      Event descriptor
    // @param[in]  x          'x' coordinate (image col)
    // @param[in]  y          'y' coordinate (images rows - row)
    // @param[in]  flags      Flags
    // @param      saver      Saver object, in this case a pointer to MouseCoordinateSaver instance 
    //
    // @note       saver is cast to void* to match callback signature required by OpenCV's setMouseCallback()
    //
    static void addCoordinates(int event, int x, int y, int flags, void* saver);

 protected:
    //--------------------------------------------------------------------------
    // pointer to target image
    const Mat*  _frame_ptr;

    //--------------------------------------------------------------------------
    // pixel coordinates
    std::vector< Point > _coordinates;
};

}  // namespace cv

#endif  // _INCLUDE_MOUSECOORDINATESAVER_HPP_