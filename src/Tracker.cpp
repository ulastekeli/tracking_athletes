#include "Tracker.h"
#include "Track.h"
#include <algorithm>
#include <limits>
#include <cmath>


Tracker::Tracker(const std::string& reid_model_path)
    : id_counter(0), frame_no(0), reid(reid_model_path) {}

void Tracker::match(std::vector<Box>& detections, const cv::Mat& frame) {
    for (auto& det : detections) {
        std::vector<int> close_tracks;
        for (int i = 0; i < tracks.size(); ++i) {
            const auto& track = tracks[i];
            int lastFrame = track.frame_history.back();
            int frameDif = frame_no - lastFrame + 1;
            const auto& box = track.box_history.back();
            float dist = std::sqrt(std::pow((box.xmin + box.xmax)/2 - (det.xmin + det.xmax)/2, 2) +
                                   std::pow((box.ymin + box.ymax)/2 - (det.ymin + det.ymax)/2, 2));
            
            if (dist <= 40 * frameDif) {
                close_tracks.push_back(i);
            }
        }
        std::cout<<close_tracks.size()<<std::endl;
        // Crop the detected object and get its feature
        det.checkBounds(frame.cols, frame.rows);
        det.print();

        cv::Rect boundingBox(det.xmin, det.ymin, det.xmax - det.xmin, det.ymax - det.ymin);
        cv::Mat cropped = frame(boundingBox);
        std::vector<float> feature = reid.run(cropped);
        if (close_tracks.empty()) {
            // Create a new track
            Track newTrack;
            newTrack.id = id_counter++;
            newTrack.update(feature, frame_no, det);
            std::cout<< " New track. ID: " << newTrack.id << " frame no: " << frame_no << std::endl;
            // Add the new track to the existing tracks
            tracks.push_back(newTrack);

        } else if (close_tracks.size() == 1) {
            // Update the track
            tracks[close_tracks[0]].update(feature, frame_no, det);

        } else {
            // Find the closest track in terms of Reid feature
            float min_dist = std::numeric_limits<float>::max();
            int best_track_idx = -1;
            for (int idx : close_tracks) {
                float dist = cv::norm(tracks[idx].features.back(), feature);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_track_idx = idx;
                }
            }

            // Update the best matching track
            tracks[best_track_idx].update(feature, frame_no, det);
        }
    }
    frame_no++;
}
