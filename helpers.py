import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment


def get_bounding_boxes(key_points, scale_factor):
    """
    Get people's bounding boxes.
    :param scale_factor: number scaling results to match original image points
    :param key_points: list of key points of all people detected in the image
    :return: list of bounding boxes (box opposite vertices) -> [left, bottom, right, top] and pose points
    """
    boxes_and_points = []
    if bool(key_points.shape):
        for n in range(key_points.shape[0]):
            points = key_points[n, :, :2]
            points_list = [tuple(scale_factor * point) for point in points]

            points = np.array(points_list, dtype=np.dtype("float, float"))
            points = points[np.nonzero(points)]
            min_x = min(points, key=lambda item: item[0])[0]
            max_x = max(points, key=lambda item: item[0])[0]
            min_y = min(points, key=lambda item: item[1])[1]
            max_y = max(points, key=lambda item: item[1])[1]

            boxes_and_points.append([[min_x, max_y, max_x, min_y], points_list])

    return boxes_and_points


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union as evaluation metric.
    :param box1: bounding box from previous frame
    :param box2: bounding box from current frame
    :return: iou score
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA + 1) * max(0, yA - yB + 1)

    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = interArea / float(box1Area + box2Area - interArea)

    return iou


def kuhn_munkres_algorithm(tracks, detections):
    """
    Kuhn-Munkres (Hungarian) Algorithm to associate the bounding boxes from previous frame
    to the bounding boxes from current frame, based on IoU metric.
    :param tracks: list of bounding boxes in previous frame
    :param detections: list of detected bounding boxes in current frame
    :return: lists of matched boxes, unmatched detections and unmatched tracks
    """
    # Generate matrix of IoU scores
    IOU_mat = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for t, trk in enumerate(tracks):
        for d, det in enumerate(detections):
            IOU_mat[t, d] = calculate_iou(trk, det)

    # Find for each detection the lowest tracking value in the matrix.
    matched_idx = linear_assignment(IOU_mat)

    # Count as unmatched detections/trackers that wasn't assigned
    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(tracks):
        if t not in matched_idx[:, 0]:
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if d not in matched_idx[:, 1]:
            unmatched_detections.append(d)

    matches = []

    for m in matched_idx:
        # If IoU score is lower than threshold, then count as unmatched
        if -IOU_mat[m[0], m[1]] < 0.1:
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
