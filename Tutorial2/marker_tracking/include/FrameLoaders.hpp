#ifndef _INCLUDE_FRAMELOADERS_HPP_
#define _INCLUDE_FRAMELOADERS_HPP_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>  // cv::VideoCapture
#include "StampedMat.hpp"      // cv::StampedMat
#include "CVUtils.hpp"         // cv::TimeStamp

//------------------------------------------------------------------------------
/// Generic class derived from OpenCV's cv::Mat class, therefore declared within cv:: namespace
///
namespace cv {

//------------------------------------------------------------------------------
/// @brief      Base abstract class describing a frame loader. Frame source and reading implemented in derived types.
///
class MultiFrameLoaderBase {
 public:
    //--------------------------------------------------------------------------
    /// @brief      Constructs a new instance.
    ///
    /// @param[in]  source  Frame source path, file (e.g. video) or directory (e.g. png images)
    ///
    explicit MultiFrameLoaderBase(const std::string& source);


    //--------------------------------------------------------------------------
    /// @brief      Destroys the object.
    ///
    virtual  ~MultiFrameLoaderBase();

    //--------------------------------------------------------------------------
    /// @brief      Number of loaded frames since last call to ::reset()
    ///
    /// @return     Frame count, unsigned integer
    ///
    size_t nFrames() const;

    //--------------------------------------------------------------------------
    /// @brief      Frame source path, file (e.g. video) or directory (e.g. png images)
    ///
    /// @return     read-only reference to member string instance
    ///
    const std::string& source() const;

    //--------------------------------------------------------------------------
    /// @brief      Loads next frame from source
    ///
    /// @note       Pure abstract, to be implemented by derived types
    ///
    /// @return     cv::StampedMat instance (time-stamped frame)
    ///
    virtual StampedMat next()  = 0;

    //--------------------------------------------------------------------------
    /// @brief      Resets frame loading, reverting to initial frame on source path
    ///
    /// @note       Pure abstract, to be implemented by derived types
    ///
    virtual void reset() = 0;

 protected:
    //--------------------------------------------------------------------------
    /// @brief      Frame counter
    ///
    size_t      _counter;

    //--------------------------------------------------------------------------
    /// @brief      Frame source path
    ///
    std::string _source;
};

//------------------------------------------------------------------------------
/// @brief      Class that reads/loads images from a directory containing individual image files (PNG format)
///             Derived from MultiFrameLoaderBase
///
class ImageLoader : public MultiFrameLoaderBase {
 public:
    //--------------------------------------------------------------------------
    /// @brief      Constructs a new instance.
    ///
    /// @param[in]  image_dir    Image directory
    /// @param[in]  file_prefix  Prefix used to filter relevant image files.
    /// @param[in]  sort         Sort flag, to be used when images contain index or time information
    ///
    explicit ImageLoader(const std::string& image_dir, const std::string& file_prefix, bool sort = true);

    //--------------------------------------------------------------------------
    /// @brief      Destroys the object.
    ///
    ~ImageLoader();

    //--------------------------------------------------------------------------
    /// @brief      Loads next image from source directory
    ///
    /// @return     cv::StampedMat instance (time-stamped frame)
    ///
    StampedMat next() override;

    //--------------------------------------------------------------------------
    /// @brief      Resets frame loading, reverting to initial frame on source path
    ///
    void reset() override;

    //--------------------------------------------------------------------------
    /// @brief      Static factory, creates a new instance (heap)
    ///
    /// @param[in]  args     Constructor arguments
    ///
    /// @tparam     ARGS_TS  Variadic template parameter pack
    //
    /// @return     Smart (shared) pointer to a new ImageLoader instance
    ///
    template< typename... ARGS_TS >
    static Ptr< ImageLoader > create(ARGS_TS... args) { return Ptr< ImageLoader > (new ImageLoader(args...)); }

    //--------------------------------------------------------------------------
    /// @brief      Extracts metadata (index, timestamp and prefix) from text string (e.g. filename and/or XML tag) in bulk
    ///             Hardcoded formatting "_", cf. implementation on source file
    ///
    /// @param[in]  text        Input text string
    /// @param[out] index_ptr   Index value extracted from [text]
    /// @param[out] stamp_ptr   Timestamp value extracted from [text]
    /// @param[out] prefix_ptr  Index value extracted from [text]
    ///
    static void getMetadata(const std::string& text, size_t* index_ptr, TimeStamp* stamp_ptr, std::string* prefix_ptr = nullptr);

    //--------------------------------------------------------------------------
    /// @brief      Extract time stamp value from input text string (e.g. filename and/or XML tag), in the format [sec]_[nsec]
    ///
    /// @note       Alternative to ::getMetadata(), with less overhead, fetch values directly instead of searching on input text.
    ///             Not as versatile.
    ///
    /// @param[in]  text      Input text string
    /// @param[in]  pos       Start position of time_stamp value on input text
    /// @param[in]  sec_size  Number of digits of the integer portion of the timestamp (number of seconds)
    ///
    /// @return     Time stamp instace with parsed value
    ///
    static TimeStamp getTimeStamp(const std::string& text, size_t pos, size_t sec_size = 10);

    //--------------------------------------------------------------------------
    /// @brief      Extract index/numeric value from input text string (e.g. filename and/or XML tag)
    ///
    /// @note       Alternative to ::getMetadata(), with less overhead, fetch values directly instead of searching on input text.
    ///             Not as versatile.
    ///
    /// @param[in]  text        Input text string
    /// @param[in]  pos         Start position of index value on input text
    /// @param[in]  index_size  The index size
    ///
    /// @return     Index value
    ///
    static size_t getIndex(const std::string& text, size_t pos, size_t index_size = 5);

    //--------------------------------------------------------------------------
    /// @brief      Compares two different text strings by the time stamp value.
    ///             Wraps around ::getMetadata() to extract time stamps from input strings
    ///
    /// @note       Useful to pass to std::sort() when sorting multiple strings (e.g. file names on source dir)
    ///
    /// @param[in]  text        Input text string
    /// @param[in]  other_text  Other input text string
    ///
    /// @return     true if [text] time stamp is greater than [other_text] time stamp
    ///
    static bool compareStamp(const std::string& text, const std::string& other_text);

    //--------------------------------------------------------------------------
    /// @brief      Lists files on the dirctory specified by [path] argument
    ///             Useful to construct list of image files to be loaded by an ImageLoader instance
    ///
    /// @note       While in this context this function is only being used to list image files (suffix defaults to ".png"),
    ///             it can be used generically to list files in a directory
    ///
    /// @param[in]  path    Path of directory to look into
    /// @param[in]  suffix  File suffix to filter down files in directory
    ///
    /// @return     Vector of text strings with the files whithin [path]
    ///
    static std::vector< std::string > listFiles(const std::string& path, const std::string& suffix = ".png");

 protected:
    //--------------------------------------------------------------------------
    /// File name prefix to identify files to be loaded
    ///
    /// @note       Useful in case of multiple trials on the same dir.
    ///
    std::string _prefix;
    //--------------------------------------------------------------------------
    /// List of image files to be loaded (within ::_path)
    ///
    /// @note       Populated/initialized at instantiation for faster image loading (no need to look at each call to ::next())
    ///
    std::vector< std::string > _files;

    //--------------------------------------------------------------------------
    /// @brief      Forward declaration of cv::FrameSynchronizer as friend class
    ///             Access to _files may be required for synchronization 
    ///
    friend class FrameSynchronizer;
};


//------------------------------------------------------------------------------
/// @brief      Class that loads frames from a video file sequentially. Wraps around OpenCV's VideoCapture class
///             Derived from MultiFrameLoaderBase
///
class VideoLoader : public MultiFrameLoaderBase {
 public:
    //--------------------------------------------------------------------------
    /// @brief      Constructs a new instance.
    ///
    /// @param[in]  video_file      Source video file
    /// @param[in]  timestamp_file  Time stamp file, with values corresponding to each frame in [video_file]
    ///
    VideoLoader(const std::string& video_file, const std::string& timestamp_file);

    //--------------------------------------------------------------------------
    /// @brief      Destroys the object.
    ///
    ~VideoLoader();

    //--------------------------------------------------------------------------
    /// @brief      Loads next frame from source video file
    ///
    /// @return     cv::StampedMat instance (time-stamped frame)
    ///
    StampedMat next() override;

    //--------------------------------------------------------------------------
    /// @brief      Resets frame loading, reverting to initial frame on video file
    ///             Releases and reloads video file
    ///
    void reset() override;

    //--------------------------------------------------------------------------
    /// @brief      Static factory, creates a new instance (heap)
    ///
    /// @param[in]  args     Constructor arguments
    ///
    /// @tparam     ARGS_TS  Variadic template parameter pack
    ///
    /// @return     Smart (shared) pointer to a new VideoLoader instance
    ///
    template< typename... ARGS_TS >
    static Ptr< VideoLoader > create(ARGS_TS... args) { return Ptr< VideoLoader > (new VideoLoader(args...)); }

    //--------------------------------------------------------------------------
    /// @brief      Loads list of timestamp values from text file
    ///
    /// @param[in]  file_name  file name
    ///
    /// @return     Vector of TimeStamp instances, corresponding to the timestamp of each frame in the video file
    ///
    static std::vector< TimeStamp > loadFile(const std::string& file_name);

 protected:
    //--------------------------------------------------------------------------
    /// VideoCapture instace that implements vid    eo decoding/frame reading
    ///
    cv::VideoCapture _video;

    //--------------------------------------------------------------------------
    /// Indexed frame time stamps
    ///
    /// @note       Size should match number of frames in the source video file, otherwise frames without a time stamp are not loaded
    ///
    std::vector< cv::TimeStamp > _stamps;

    //--------------------------------------------------------------------------
    /// @brief      Forward declaration of cv::FrameSynchronizer as friend class
    ///             Access to _stamps may be required for synchronization 
    ///
    friend class FrameSynchronizer;
};

}  // namespace cv

#endif  // _INCLUDE_FRAMELOADERS_HPP_

