#ifndef _INCLUDE_FRAMESYNCHRONIZER_HPP_
#define _INCLUDE_FRAMESYNCHRONIZER_HPP_

#include <string>
#include <vector>
#include <cstdarg>
#include <initializer_list>
#include "FrameLoaders.hpp"
//------------------------------------------------------------------------------
/// @brief      Generic use (in parallel with MultiFrameLoaderBase and derived), therefore declared within cv::
///
namespace cv {
//------------------------------------------------------------------------------
/// @brief      Class that synchronizes frames from multiple sources/loaders (derived from MultiFrameLoaderBase)
///
class FrameSynchronizer {
 public:
    //--------------------------------------------------------------------------
    /// @brief      Loader type alias, declared for convenience/readability
    typedef cv::Ptr< MultiFrameLoaderBase > BaseLoaderPtr;

    //--------------------------------------------------------------------------
    /// @brief      Creates a new instance from single time threshold value
    ///
    /// @param[in]  stamp_threshold  Default time threshold for frames to be considered synchronous (defaults to 1ms)
    ///
    explicit FrameSynchronizer(cv::TimeStamp stamp_threshold = static_cast< cv::TimeStamp >(0.001));

    //--------------------------------------------------------------------------
    /// @brief      Creates a new instance from a list of ::BaseLoaderPtr objects
    ///
    /// @param[in]  loaders          List ("{...}") of cv::Ptr objects managing any type derived from MultiFrameLoaderBase
    /// @param[in]  stamp_threshold  Default time threshold for frames to be considered synchronous (defaults to 1ms)
    ///
    explicit FrameSynchronizer(const std::initializer_list< BaseLoaderPtr >& loaders, cv::TimeStamp stamp_threshold = static_cast< cv::TimeStamp >(0.001));

    //--------------------------------------------------------------------------
    /// @brief      Creates a new instance from a std::vector of ::BaseLoaderPtr objects
    ///             Useful if a std::vector instance is already managing multiple loaders
    ///
    /// @param[in]  loaders          Vector of cv::Ptr objects managing any type derived from MultiFrameLoaderBase
    /// @param[in]  stamp_threshold  Default time threshold for frames to be considered synchronous (defaults to 1ms)
    ///
    explicit FrameSynchronizer(const std::vector< BaseLoaderPtr >& loaders, cv::TimeStamp stamp_threshold = static_cast< cv::TimeStamp >(0.001));

    //--------------------------------------------------------------------------
    /// @brief      Creates a new instance from multiple frame loaders
    ///
    /// @param[in]  loaders      Multiple (comma-separated) frame loader instances
    ///
    /// @tparam     LoaderTypes  Variadic parameter pack encompassing different loader types (derived from MultiFrameLoaderBase)
    ///
    template< typename... LoaderTypes >
    FrameSynchronizer(const LoaderTypes&... loaders) : FrameSynchronizer({ loaders... }) { /* ... */}

    //--------------------------------------------------------------------------
    /// @brief      Frame loaders being synchronized by the instance.
    ///
    /// @note       Useful when creating a new FrameSynchronizer instance for the same frame loader objects (e.g. different threshold)
    ///
    /// @return     Read-only reference to member vector holding cv::Ptr instances for each loader object
    ///
    const std::vector< BaseLoaderPtr >& loaders() const;

    //--------------------------------------------------------------------------
    /// @brief      Adds a new loader to be synchronized
    ///
    /// @param[in]  Frame loader
    ///
    void add(const BaseLoaderPtr& loader);

    //--------------------------------------------------------------------------
    /// @brief      Load new frames
    ///
    /// @param      frames  Vector of cv::StampedMat instances to assign new frames to
    ///
    /// @return     Loading/assignment success status
    ///
    bool operator>>(std::vector< StampedMat >& frames);

    //--------------------------------------------------------------------------
    /// @brief      Load new frames using default (member) threshold
    ///
    /// @return     Vector of cv::StampedMat instances with new frames
    ///
    std::vector< StampedMat > next();

    //--------------------------------------------------------------------------
    /// @brief      Load new frames bypassing default (member) threshold (using stamp_threshold)
    ///
    /// @param[in]  stamp_threshold  Time threshold for frames to be considered synchronous
    ///
    /// @return     Vector of cv::StampedMat instances with new frames
    ///
    std::vector< StampedMat > next(double stamp_threshold);

    //--------------------------------------------------------------------------
    /// @brief      Synchronizes frames loaded from each loader
    ///
    /// @note       Static member in order to allow one-time syncrhonization of different loaders without instantiaton ('bare')
    ///
    /// @param      loaders          Vector of frame loaders
    /// @param[in]  stamp_threshold  Time threshold for frames to be considered synchronous
    ///
    /// @return     Vector of cv::StampedMat instances with new frames
    ///
    static std::vector< StampedMat > syncNextFrames(std::vector< BaseLoaderPtr >& loaders, cv::TimeStamp stamp_threshold = static_cast< cv::TimeStamp >(0.001));

    static std::vector< StampedMat > syncNextFrames2(std::vector< BaseLoaderPtr >& loaders, cv::TimeStamp stamp_threshold = static_cast< cv::TimeStamp >(0.001));

 protected:
    //--------------------------------------------------------------------------
    /// Time threshold for frames to be considered synchronous
    /// Default value used when ::next() or ::operator>>() are called
    ///
    double _threshold;

    //--------------------------------------------------------------------------
    /// Vector of cv::Ptr instances holding frame loaders derived from MultiFrameLoaderBase
    ///
    std::vector< BaseLoaderPtr > _loaders_ptrs;
};
///////////////////////////////////////////////////////////////////////////
}  // namespace cv
///////////////////////////////////////////////////////////////////////////
#endif  // _INCLUDE_FRAMESYNCHRONIZER_HPP_