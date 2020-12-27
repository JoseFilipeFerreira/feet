#include <dirent.h>            // DIR, opendir, closedir
#include <vector>
#include <exception>
#include <string>
#include <fstream>
#include <iostream>
#include "FrameLoaders.hpp"
#include "StampedMat.hpp"

/////////////////////////////////////////////////////////////
namespace cv {
/////////////////////////////////////////////////////////////
MultiFrameLoaderBase::MultiFrameLoaderBase(const std::string& source) :
    _source(source),
    _counter(0) {
        /* ... */
}
MultiFrameLoaderBase::~MultiFrameLoaderBase() {
    /* ... */
}
size_t MultiFrameLoaderBase::nFrames() const {
    return _counter;  // number of frames loaded (since last reset)
}
/////////////////////////////////////////////////////////////
ImageLoader::ImageLoader(const std::string& image_dir, const std::string& file_prefix, bool sort):  // , const std::string& file_name_format) :
    MultiFrameLoaderBase(image_dir), 
    _prefix(file_prefix) {
        // looks for png files on 'image_dir/'
        _files = listFiles(image_dir, ".png");
        // sort vector according to timestamp
        // alternatively, sort alphabetically, but that's overkill and add overhead
        // NOTE: sort == false should only be passed if one is sure files are loaded in the desired order
        if (sort == true) {
            std::sort(_files.begin(), _files.end(), ImageLoader::compareStamp);
        }
        // check result
        if (!_files.size()) {
            throw std::invalid_argument("ImageLoader::ImageLoader() -> Empty source directory;");
        }
        // initialize _counter and open video file
        reset();
}
ImageLoader::~ImageLoader() {
    /* ... */
}
StampedMat ImageLoader::next() {
    // fetch frame index and time stamp from file name
    // uses member format string
    // TODO(joao): do something with prefix -> compare with member to filter files for a specific prefix
    // NOTE: index != _counter
    // size_t    index = 0;
    // TimeStamp stamp = 0.0;
    // load image data
    if (_counter >= _files.size()) {
        reset();
        throw std::runtime_error("ImageLoader::next(): no more frames!");
    }
    std::string path = _source + "/" +_files[_counter];
    StampedMat frame;
    try {
        frame = cv::imread(path.data(), IMREAD_ANYDEPTH);
        // frame = cv::imread(path.data(), CV_LOAD_IMAGE_ANYDEPTH);  // OpenCV 3.X.X
    } catch (cv::Exception& error) {
        throw std::runtime_error("ImageLoader::next(): unable to load image;");
    }
    if (frame.empty()) {
        // std::cout << _counter << std::endl;
        throw std::runtime_error("ImageLoader::next(): empty frame;");
    }
    // assign metada in-place
    try {
        frame.index = getIndex(_files[_counter], _prefix.size() + 1);
        frame.stamp = getTimeStamp(_files[_counter], _prefix.size() + 7);
    } catch (std::invalid_argument& error) {
        throw std:: runtime_error("ImageLoader::next(): unable to fetch metada from file name;");
    }
    // increment counter
    _counter++;
    // 
    return frame;
}
void ImageLoader::reset() {
    // back to first frame
    _counter = 0;
}
// static utility function
// parses string according to format to extract 
// NOTE: _prefix is not being used atm
void ImageLoader::getMetadata(const std::string& text, size_t* index_ptr, TimeStamp* stamp_ptr, std::string* prefix_ptr) {  //, const std::string& format) {
    // NOTE: sscanf not working, discarded atm
    // if (sscanf(text.data(), format.data(), index_ptr, stamp_sec_ptr, stamp_nsec_ptr) < 3) {
    //     throw std::invalid_argument("ImageLoader::getMetadata() -> Unable to parse input text (check naming format);");
    // }
    std::vector< std::string > strings;
    std::istringstream         f(text);
    std::string                s;
    // split input string into substrings separated by '_'
    while (getline(f, s, '_')) {
        // std::cout << s << std::endl;
        strings.push_back(s);
    }
    // assign metadata to last 3 substrings
    // NOTE: assumes a fixed/formatted file name
    if (index_ptr != nullptr) {
        *index_ptr     = atoi(strings[strings.size() - 3].data());
    }
    if (stamp_ptr != nullptr) {
        int stamp_sec  = atoi(strings[strings.size() - 2].data());
        int stamp_nsec = atoi(strings[strings.size() - 1].data());
        *stamp_ptr     = static_cast< TimeStamp >(stamp_sec + stamp_nsec * 0.000000001);  // NOTE: direct assignmnt may not work if ros::Time is being used
    }
    // construct prefix string by adding remaining substrings
    // maybe not the most efficient way to do this (input string is truncated and then)
    if (prefix_ptr != nullptr) {
        for (int i = 0; i < strings.size() - 3; ++i) {
            *prefix_ptr += strings[i];
        }
    }
}
// static metada acessors
// less overhead, positions need to be known beforehand
TimeStamp ImageLoader::getTimeStamp(const std::string& text, size_t pos, size_t sec_size) {
    if (text.size() < (pos + sec_size + 1 /* sep*/ + 9 /* nsec*/ )) {
        throw std::invalid_argument("ImageLoader::getTimeStamp(): input text too short;");
    } else {
        return static_cast< TimeStamp >(atoi(text.substr(pos, sec_size).data()) + atoi(text.substr(pos + sec_size + 1, 9).data()) * 0.000000001);
    }
}
size_t ImageLoader::getIndex(const std::string& text, size_t pos, size_t index_size) {
    if (text.size() < (pos + index_size)) {
        throw std::invalid_argument("ImageLoader::getIndex(): input text too short;");
    } else {
        return atoi(text.substr(pos, index_size).data());
    }
}
// static comparison function
// compares filenames according to time stamps on file name
// useful to sort file list according to timestamp
// TODO(joao): maybe declare fucntion outside ImageLoader class??
bool ImageLoader::compareStamp(const std::string& text, const std::string& other_text) {
    TimeStamp stamp, other_stamp;
    // parse file names
    // NOTE: _prefix is passed but not used
    // NOTE: index and other_index are populatd but not used
    getMetadata(text, nullptr, &stamp);
    getMetadata(other_text, nullptr, &other_stamp);
    // return comparison result
    // NOTE: assumes comparison operator ('<') is defined for cv::TimeStamp type (typedef, cf. StampedMat.hpp)
    return (stamp < other_stamp);
}
// list files and dirs on a path, filters by suffix
std::vector< std::string > ImageLoader::listFiles(const std::string& path, const std::string& suffix) {
    std::vector< std::string > files;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (path.data())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string entry(ent->d_name);
            // search for suffix in file/dir name
            if (entry.find(suffix) != std::string::npos) {
                // if found, append to file name vector
                files.emplace_back(ent->d_name);
                // std::cout << ent->d_name << std::endl;
            }
        }
        closedir(dir);
    } else {
        throw std::runtime_error("listFiles() -> could not open path;");
    }
    return files;
}
/////////////////////////////////////////////////////////////
VideoLoader::VideoLoader(const std::string& video_file, const std::string& timestamp_file) :
    MultiFrameLoaderBase(video_file) {
        // load time stamps
        _stamps = loadFile(timestamp_file);
        // initialize _counter and open video file
        reset();
}
VideoLoader::~VideoLoader() {
    /* ... */
}
StampedMat VideoLoader::next() {
    // check index before fetching frame
    if (_counter >= _stamps.size()) {
        reset();
        throw std::runtime_error("VideoLoader::next(): no more frames!");
        // if (_loop) {
        //     reset();  // _counter is set to 0
        // } else {
        //     throw std::runtime_error("VideoLoader::next() -> last frame reached;");
        // }
    }
    // load frame from video file
    StampedMat frame;
    _video >> frame;
    // check frame/index validity
    if (frame.empty()) {
        if (_counter < _stamps.size()) {
            // throw std::runtime_error("VideoLoader::next() -> invalid/missing frame from video file (< _stamps.size())");
            // if more stamps than frames in video, remove remaining stamps
            _stamps.erase(_stamps.begin() + _counter, _stamps.end());
        }
    } else {
        // valid frame, but missing timestamp? 
        // never reached, will reset or throw exception on lines 127/125
        if (_counter > _stamps.size()) {
            throw std::runtime_error("VideoLoader::next() -> missing timestamp values (>= _stamps.size())");
            // cv::waitKey(0);
            // _stamps.emplace_back(-1.0);
        }
    }
    // assign metadata
    frame.index = _counter - 1;
    frame.stamp = _stamps[_counter];
    // increment counter 
    _counter++;
    // instantiate on return
    return frame;
}
void VideoLoader::reset() {
    _counter = 0;
    _video.open(_source);
}
// load list of doubles from file
std::vector< TimeStamp > VideoLoader::loadFile(const std::string& file_name) {
    // std::string value_str;
    std::vector< TimeStamp> values;
    // std::stringstream stream;
    std::ifstream file(file_name.data(), std::ios::in);
    file.precision(20);
    // int l = 0;
    // int v = 0;
    TimeStamp value = 0.0;
    // std::cout << "file: " << file_name << std::endl;
    if (file.is_open()){
        while (file >> value) {
            // std::cout << std::setprecision(19) << value << std::endl;
            values.push_back(value);
        }    
        // while (getline(file, value_str) ){
        //     // l++;
        //     stream.str(value_str);
        //     values.push_back(std::stod(stream.str()));
        //     stream.clear();
        // }
        file.close();
    }
    return values;
};
/////////////////////////////////////////////////////////////
}  // namespace cv
/////////////////////////////////////////////////////////////