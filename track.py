import cv2
import numpy as np
from keypoints import KeyPointDetection
from kalmantracker import Tracker
from helpers import get_bounding_boxes, kuhn_munkres_algorithm


class Tracking:
    def __init__(self):
        # Stores all active trackers
        self.tracker_list = []
        # Max number of frames without matched track
        self.max_age = 12
        # Next id to assign
        self.next_id = 1

    def track(self, img):
        """
        Main function to handle tracking problem.
        :param img: Image resized by scale factor.
        :return: Image with skeletons and id numbers drawn.
        """
        # Decrease image size to get speed improvement
        scale_factor = 4
        resized_image = cv2.resize(frame, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_LINEAR)

        # Call detection methods to get bounding boxes and pose points in current frame
        key_points = key_point_detector.find_key_points(resized_image)
        boxes_and_points = get_bounding_boxes(key_points, scale_factor)

        key_points_lists = [item[1] for item in boxes_and_points]
        z_boxes = [item[0] for item in boxes_and_points]
        prev_boxes = [tracker.bounding_box for tracker in self.tracker_list]

        # Use Hungarian Algorithm to find matches between frames
        matches, unmatched_detections, unmatched_tracks = kuhn_munkres_algorithm(prev_boxes, z_boxes)

        # If detection was matched to track, use both phases of Kalman filter.
        for t, d in matches:
            self.tracker_list[t].predict()
            self.tracker_list[t].update(np.expand_dims(z_boxes[d], axis=0).T)
            self.tracker_list[t].bounding_box = list(self.tracker_list[t].x.T[0][:4])
            self.tracker_list[t].key_points = key_points_lists[d]
            self.tracker_list[t].matched_detection = True
            self.tracker_list[t].unmatched_tracks = 0

        # If detection wasn't matched, create new tracker for new person.
        for d in unmatched_detections:
            z = np.expand_dims(z_boxes[d], axis=0).T
            x = np.array([[z[0], z[1], z[2], z[3], 0, 0, 0, 0]]).T
            new_tracker = Tracker(x, list(x.T[0][:4]), key_points_lists[d], self.next_id)
            self.next_id += 1
            self.tracker_list.append(new_tracker)

        # If track wasn't matched, call predict phase and count unmatched frame.
        for t in unmatched_tracks:
            self.tracker_list[t].predict()
            self.tracker_list[t].unmatched_tracks += 1
            self.tracker_list[t].bounding_box = list(self.tracker_list[t].x.T[0][:4])

        # Draw bounding boxes and id numbers.
        for tracker in self.tracker_list:
            if tracker.matched_detection and tracker.unmatched_tracks <= self.max_age:
                key_point_detector.draw_skeleton(tracker.key_points, img)
                cv2.putText(img, f"id={tracker.person_id}",
                            (int(tracker.bounding_box[0]), int(tracker.bounding_box[3]) - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

        # Delete all trackers with number of unmatched tracks greater than max age of tracker.
        self.tracker_list = [tracker for tracker in self.tracker_list if tracker.unmatched_tracks <= self.max_age]

        return img


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    key_point_detector = KeyPointDetection()
    tracking = Tracking()

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        output = tracking.track(frame)
        cv2.imshow("Detected", output)
        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
