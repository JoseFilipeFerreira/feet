#include "FrameSynchronizer.hpp"
#include <vector>
///////////////////////////////////////////////////////////////////////////
namespace cv {
///////////////////////////////////////////////////////////////////////////
// FrameSynchronizer::FrameSynchronizer(const std::string& rgb_path,
//                                      const std::string& depth_path,
//                                      const std::string& rgb_stamp_file,
//                                      const std::string& depth_prefix,
//                                      bool               loop) :
//     _rgb_loader(rgb_path, rgb_stamp_file, loop),
//     _depth_loader(depth_path, depth_prefix, loop, true /* force bulk sort */) {
//         /* ... */
// }
// FrameSynchronizer::FrameSynchronizer(VideoLoader* rgb_loader_ptr, ImageLoader* depth_loader_ptr) :
//     _rgb_loader(*rgb_loader_ptr),
//     _depth_loader(*depth_loader_ptr) {
//         /* ... */
// }
// ///////////////////////////////////////////////////////////////////////////
// // gets matched frames
// std::vector< StampedMat > FrameSynchronizer::next(double stamp_threshold) {
//     std::vector< StampedMat > frames;
//     // gets next color frame
//     frames.emplace_back(_rgb_loader.next());
//     // gets next depth frame
//     frames.emplace_back(_depth_loader.next());
//     while (abs(frames[DEPTH].stamp() - frames[RGB].stamp()) > stamp_threshold) {
//         frames[DEPTH] = _depth_loader.next();
//     }
//     return frames;
// }
///////////////////////////////////////////////////////////////////////////
FrameSynchronizer::FrameSynchronizer(cv::TimeStamp stamp_threshold) :
    _threshold(stamp_threshold) {
        /* ... */
}
FrameSynchronizer::FrameSynchronizer(const std::initializer_list< FrameSynchronizer::BaseLoaderPtr >& loaders, cv::TimeStamp stamp_threshold) :
    _threshold(stamp_threshold),
    _loaders_ptrs(loaders) {
        /* ... */
}
FrameSynchronizer::FrameSynchronizer(const std::vector< FrameSynchronizer::BaseLoaderPtr >& loaders, cv::TimeStamp stamp_threshold) :
    _threshold(stamp_threshold),
    _loaders_ptrs(loaders) {
        /* ... */
}
// acess to underlying loader ptrs
const std::vector< FrameSynchronizer::BaseLoaderPtr >& FrameSynchronizer::loaders() const {
    return _loaders_ptrs;
}
// add new loader individually
void FrameSynchronizer::add(const BaseLoaderPtr& loader) {
    _loaders_ptrs.emplace_back(loader);
}
// frame accessors/incrementers
bool FrameSynchronizer::operator>>(std::vector< StampedMat >& frames) {
    frames = syncNextFrames(_loaders_ptrs, _threshold);
    return true;
}
std::vector< StampedMat > FrameSynchronizer::next() {
    return syncNextFrames(_loaders_ptrs, _threshold);
}
std::vector< StampedMat > FrameSynchronizer::next(double stamp_threshold) {
    return syncNextFrames(_loaders_ptrs, stamp_threshold);
}
// frame synchronization function
// NOTE: faster if frame loaders are sorted, otherwise it must loop back
// the issue is that there is no guarantee that independent frames have similar stamps, and an adaptive solution to find the closest frames would be too heavy
std::vector< StampedMat> FrameSynchronizer::syncNextFrames(std::vector< FrameSynchronizer::BaseLoaderPtr >& loaders_ptrs, cv::TimeStamp stamp_threshold) {
    std::vector< StampedMat > frames;
    // gets next frame for each loader
    // store max timestamp to use as reference
    // NOTE: best case scenario frames are sorted by stamp, and remaining loaders have to catch up
    //       worst case scenario frames are not sorted, and we will have to loop over all frames -> great overhead
    // printf("-----------------------------------------------------------------\nThreshold: %f \nSelecting reference (greater time stamp):\n", stamp_threshold);
    uint reference_idx = 0;
    std::cout.precision(4);
    for (int idx = 0;  idx < loaders_ptrs.size(); idx++) {
        frames.emplace_back(loaders_ptrs[idx]->next());
        // std::cout << std::fixed << frames[idx].stamp;
        if (frames[idx].stamp > frames[reference_idx].stamp) {
            // printf(" ahead of reference!");
            reference_idx = idx;
        }
        // std::cout << std::endl;
    }
    // std::cout << "reference: " << reference_idx << std::endl;
    // reloop over frame loaders and increment remaining loaders until they match reference loader
    for (int idx = 0;  idx < loaders_ptrs.size(); idx++) {
        // skip if reference loader, for efficiency
        // in practical terms it would never increment frame on reference load as stamps would match
        if (idx == reference_idx) {
            continue;
        }
        // std::cout << "matching #" << idx << " with reference stamp " << frames[reference_idx].stamp << std::endl;
        while (abs(frames[idx].stamp - frames[reference_idx].stamp) > stamp_threshold) {
            frames[idx] = loaders_ptrs[idx]->next();
            // std::cout  << std::fixed << frames[idx].stamp << " // " <<  frames[reference_idx].stamp - frames[idx].stamp << " [" << stamp_threshold << "]" << std::endl;
            if (frames[idx].stamp > frames[reference_idx].stamp) {
                break;
            }
        }
        // std::cout << "\n new ts: " << std::fixed << frames[idx].stamp << std::endl;
        if (abs(frames[idx].stamp - frames[reference_idx].stamp) > stamp_threshold) {
            throw std::runtime_error("FrameSynchronizer::syncNextFrames(): unable to syncronize!");
        }
    }
    // printf("-----------------------------------------------------------------\nDebug DONE\n");
    return frames;
}

// alternative approach
std::vector< StampedMat> FrameSynchronizer::syncNextFrames2(std::vector< FrameSynchronizer::BaseLoaderPtr >& loaders_ptrs, cv::TimeStamp stamp_threshold) {
    std::vector< StampedMat > frames;
    // use smaller time stamp (frame that is behind) as reference
    std::vector< cv::TimeStamp > stamps;
    for (const auto& loader : loaders_ptrs) {
        try {
            VideoLoader& derived = dynamic_cast< VideoLoader& >(*loader);
            stamps.emplace_back(derived._stamps[derived._counter]);  // @note       FrameSynchronizr is a friend class
        } catch (std::exception&) {
            try {
                ImageLoader& derived = dynamic_cast< ImageLoader& >(*loader);
                stamps.emplace_back(ImageLoader::getTimeStamp(derived._files[derived._counter], derived._prefix.size() + 7));  // @note FrameSynchronizr is a friend class
            } catch (std::exception&) {
                throw std::runtime_error("syncNextFrames2(): invalid loader type");
            }
        }
    }

    std::cout << stamps.size() << std::endl;

    cv::TimeStamp min_stamp = stamps[0];
    for (int idx = 1; idx < stamps.size(); idx++) {
        if (stamps[idx] < min_stamp) {

        }
    }


    // gets next frame for each loader
    // store max timestamp to use as reference
    // NOTE: best case scenario frames are sorted by stamp, and remaining loaders have to catch up
    //       worst case scenario frames are not sorted, and we will have to loop over all frames -> great overhead
    // printf("-----------------------------------------------------------------\nThreshold: %f \nSelecting reference (greater time stamp):\n", stamp_threshold);
    uint reference_idx = 0;
    std::cout.precision(4);
    for (int idx = 0;  idx < loaders_ptrs.size(); idx++) {
        // frames.emplace_back(loaders_ptrs[idx]->next());
        // std::cout << std::fixed << frames[idx].stamp;
        if (frames[idx].stamp < frames[reference_idx].stamp) {
            // printf(" ahead of reference!");
            reference_idx = idx;
        }
        // std::cout << std::endl;
    }
  
    return frames;
}



///////////////////////////////////////////////////////////////////////////
}  // namespace cv


