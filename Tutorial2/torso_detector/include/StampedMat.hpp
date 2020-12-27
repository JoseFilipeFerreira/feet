#ifndef _INCLUDE_CV_STAMPEDMAT_HPP_
#define _INCLUDE_CV_STAMPEDMAT_HPP_

#include <opencv2/opencv.hpp>  // cv::Mat
#include "CVUtils.hpp"         // cv::TimeStamp

//------------------------------------------------------------------------------
/// Generic class derived from OpenCV's cv::Mat class, therefore declared within cv:: namespace
///
namespace cv {
//------------------------------------------------------------------------------
/// @brief      Class that describes a common OpenCV array with index and time stamp members.
///             Directly derived from Mat, as such compatible with any function accepting cv::Mat
///
class StampedMat : public Mat {
 public:
    //--------------------------------------------------------------------------
    /// @brief      Empty constructor, instantiates a new empty StampedMat object
    ///             Index and time stamp initialized to null values
    ///
    StampedMat() : index(0), stamp(0.0) { /* ... */ }

    //--------------------------------------------------------------------------
    /// @brief      Copy construtor, instantiates a new StampedMat object from an existing cv::Mat
    ///             Index and time stamp initialized to null values
    ///
    /// @param[in]  other  Instance to copy
    ///
    StampedMat(const Mat& other) : index(0), stamp(0.0), Mat(other) { /* ... */ }

    //--------------------------------------------------------------------------
    /// @brief      Advanced initialization constructor, assigns index and time stamp to argument values
    ///             Wraps around Mat constructors (variable argument number/type)
    ///
    /// @note       It is not possible to use default values for index/timestamp or a single variadic parameter pack
    ///             as it would cause ambiguity with other constructors
    ///
    /// @param[in]  index   Frame index
    /// @param[in]  stamp   Frame time stamp (cf. TimeStamp definition)
    /// @param[in]  args    Initialization arguments (forwarded to base class constructor)
    ///
    /// @tparam     ARG_TS  Variadic template parameter pack
    ///
    template < typename... ARG_TS>
    StampedMat(size_t index, const TimeStamp& stamp, ARG_TS... args) : index(index), stamp(stamp), Mat(args...) { /* ... */ }

    //--------------------------------------------------------------------------
    /// @brief      Creates a new instance (heap). Wraps around cv::Mat constructors (variable argument number/type)
    ///
    /// @param[in]  args    Initialization arguments (forwarded to base class constructor)
    ///
    /// @tparam     ARG_TS  Variadic template parameter pack
    ///
    /// @return     Smart (shared) pointer to a new StampedMat instance
    ///
    template < typename... ARG_TS >
    static cv::Ptr< StampedMat > create(ARG_TS... args) { return cv::Ptr< StampedMat >(args...); }

    //--------------------------------------------------------------------------
    /// frame index / unique identifier value
    ///
    size_t index;

    //--------------------------------------------------------------------------
    /// frame time stamp value
    ///
    /// @note       cf. cv::TimeStamp (CVUtils.hpp)
    ///
    TimeStamp stamp;
};

}  // namespace cv

#endif  //  _INCLUDE_CV_STAMPEDMAT_HPP_
