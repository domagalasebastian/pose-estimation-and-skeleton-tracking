import cv2
from openpose import pyopenpose as op


class KeyPointDetection:
    def __init__(self):
        """
        Init and config OpenPose wrapper for Python.
        """
        self.params = dict()
        # Path to folder with models
        self.params["model_folder"] = "models"
        # Set net resolution, lower resolution -> faster, lower accuracy
        self.params["net_resolution"] = "-1x224"
        # Select model, BODY_25 is the most accurate and the fastest
        self.params["model_pose"] = "BODY_25"
        # Init wrapper
        self.datum = op.Datum()
        self.wrapper = op.WrapperPython()
        # Configure wrapper
        self.wrapper.configure(self.params)
        self.wrapper.start()
        # List of pairs to draw a skeleton
        self.pose_pairs = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6],
                           [6, 7], [0, 15], [15, 17], [0, 16], [16, 18], [1, 8],
                           [8, 9], [9, 10], [10, 11], [11, 22], [22, 23], [11, 24],
                           [8, 12], [12, 13], [13, 14], [14, 19], [19, 20], [14, 21]]

    def find_key_points(self, image):
        """
        Detect people's key points in the image.
        :param image: decreased frame
        :return: array of key points in shape (num_of_detected_skeletons, 25, 3) - > [x, y, probability]
        """
        self.datum.cvInputData = image
        self.wrapper.emplaceAndPop([self.datum])

        return self.datum.poseKeypoints

    def draw_skeleton(self, key_points, img):
        """
        Use detected key points to draw skeleton of a person.
        :param key_points: key points of a person
        :param img: frame in original size
        :return: frame in original size with skeletons drawn
        """
        for pair in self.pose_pairs:
            partA = pair[0]
            partB = pair[1]

            if all((key_points[partA][0], key_points[partA][1], key_points[partB][0], key_points[partB][1])):
                cv2.line(img, key_points[partA], key_points[partB], (212, 255, 127), 3)
                cv2.circle(img, key_points[partA], 5, (255, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(img, key_points[partB], 5, (255, 255, 255), thickness=-1, lineType=cv2.FILLED)

        return img
