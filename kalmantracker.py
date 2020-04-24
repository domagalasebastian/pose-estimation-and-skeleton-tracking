from scipy.linalg import inv
import numpy as np


class Tracker:
    def __init__(self, x, bounding_box, key_points, person_id):
        """
        Init single tracker for detected person.
        """
        self.person_id = person_id
        self.key_points = key_points
        self.bounding_box = bounding_box
        self.matched_detection = False
        self.unmatched_tracks = 0

        # Kalman filter parameters
        self.x = x
        self.dT = 1
        self.F = np.array([[1, 0, 0, 0, self.dT, 0, 0, 0],
                           [0, 1, 0, 0, 0, self.dT, 0, 0],
                           [0, 0, 1, 0, 0, 0, self.dT, 0],
                           [0, 0, 0, 1, 0, 0, 0, self.dT],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1]])
        self.G = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0]])

        self.P = np.diag(10.0 * np.ones(8))
        self.Q = np.diag(0.01 * np.ones(4))
        self.R = np.identity(4)
        self.R[2, 2] = 10.0
        self.R[3, 3] = 10.0

    def predict(self):
        """
        Predict phase of Kalman filter.
        """
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.G.dot(self.Q).dot(self.G.T)

    def update(self, z):
        """
        Update phase of Kalman filter.
        :param z: measurement - bounding box from detection
        """
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(inv(S))
        e = z - self.H.dot(self.x)
        self.x += K.dot(e)
        self.P -= K.dot(self.H).dot(self.P)
