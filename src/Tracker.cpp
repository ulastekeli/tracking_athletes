#include "Tracker.h"
#include <algorithm>
#include <limits>
#include <cmath>

Tracker::Tracker(const std::string& reid_model_path)
    : id_counter(0), frame_no(0), reid(reid_model_path) {}

void Tracker::match(const std::vector<DetectedObject>& detections, const cv::Mat& frame) {
    frame_no++;
    for (const auto& det : detections) {
        std::vector<int> close_tracks;
        for (int i = 0; i < tracks.size(); ++i) {
            const auto& track = tracks[i];
            const auto& box = track.box_history.back();
            float dist = std::sqrt(std::pow(box.xmin + box.xmax/2 - det.boundingBox.x - det.boundingBox.width/2, 2) +
                                   std::pow(box.ymin + box.ymax/2 - det.boundingBox.y - det.boundingBox.height/2, 2));
            if (dist <= 30) {
                close_tracks.push_back(i);
            }
        }
        // Crop the detected object and get its feature
        cv::Mat cropped = frame(detection.boundingBox);
        std::vector<float> feature = reid.run(cropped);
        if (close_tracks.empty()) {
            // Create a new track
            Box newBox(detection.boundingBox.x, detection.boundingBox.y, detection.boundingBox.width, detection.boundingBox.height, 1.0);
            Track newTrack;
            newTrack.id = id_counter++;
            newTrack.update(feature, frame_no, newBox);

            // Add the new track to the existing tracks
            tracks.push_back(newTrack);

        } else if (close_tracks.size() == 1) {
            // Update the track
            Box newBox(detection.boundingBox.x, detection.boundingBox.y, detection.boundingBox.width, detection.boundingBox.height, 1.0);
            tracks[close_tracks[0]].update(feature, frame_no, newBox);

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
            Box newBox(detection.boundingBox.x, detection.boundingBox.y, detection.boundingBox.width, detection.boundingBox.height, 1.0);
            tracks[best_track_idx].update(feature, frame_no, newBox);
        }
    }
}
